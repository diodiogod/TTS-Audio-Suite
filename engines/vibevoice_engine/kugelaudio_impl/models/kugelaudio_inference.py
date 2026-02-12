"""KugelAudio inference model for speech generation.

This is the open-source inference implementation without optimizations.
Based on the original VibeVoice model architecture.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import modeling_utils
from transformers.cache_utils import DynamicCache
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.utils import logging

from ..configs import KugelAudioConfig
from ..schedule.dpm_solver import DPMSolverMultistepScheduler
from .diffusion_head import KugelAudioDiffusionHead
from .kugelaudio_model import KugelAudioModel, KugelAudioPreTrainedModel
from .tokenizer import (
    KugelAudioTokenizerEncoderOutput,
    KugelAudioTokenizerStreamingCache,
)

logger = logging.get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


def _get_cache_tensors(cache) -> Tuple[List, List]:
    """Get key and value cache tensors from a cache object."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache, cache.value_cache
    raise AttributeError(f"Cannot get cache tensors from {type(cache).__name__}")


@dataclass
class KugelAudioCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class KugelAudioGenerationOutput(ModelOutput):
    """Output type for KugelAudio generation."""

    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None


class KugelAudioTokenConstraintProcessor(LogitsProcessor):
    """Constrains token generation to only valid tokens during speech generation."""

    def __init__(self, valid_token_ids: List[int], device: torch.device = None):
        self.valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.valid_token_ids] = 0
        scores = scores + mask
        return scores


class KugelAudioForConditionalGenerationInference(KugelAudioPreTrainedModel, GenerationMixin):
    """KugelAudio model for inference with speech generation capabilities."""

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = KugelAudioModel(config)
        self.lm_head = nn.Linear(
            config.decoder_config.hidden_size,
            config.decoder_config.vocab_size,
            bias=False,
        )
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps
        self.post_init()

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head

    @property
    def speech_scaling_factor(self):
        return self.model.speech_scaling_factor

    @property
    def speech_bias_factor(self):
        return self.model.speech_bias_factor

    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer

    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector

    @property
    def semantic_tokenizer(self):
        return self.model.semantic_tokenizer

    @property
    def semantic_connector(self):
        return self.model.semantic_connector

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = (
            num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps
        )

    def _process_speech_inputs(
        self,
        speech_tensors: Optional[torch.Tensor] = None,
        speech_masks: Optional[torch.Tensor] = None,
        voice_cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process speech inputs through acoustic and semantic encoders.

        Supports two paths:
        1. voice_cache: Pre-encoded voice features (from .pt files or encode_voice_prompt)
        2. speech_tensors: Raw audio tensor that gets encoded through both tokenizers

        Returns:
            Tuple of (acoustic_features, speech_embeds) where speech_embeds has shape
            [num_valid_frames, hidden] - already indexed by speech_masks for direct
            assignment to inputs_embeds[speech_input_mask].
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if voice_cache is not None:
            # Use pre-encoded voice features
            acoustic_mean = voice_cache["acoustic_mean"].to(device=device, dtype=dtype)
            semantic_mean = voice_cache.get("semantic_mean")

            # Sample from acoustic distribution
            fix_std = voice_cache.get("acoustic_std", self.acoustic_tokenizer.fix_std)
            acoustic_features = acoustic_mean + fix_std * torch.randn_like(acoustic_mean)

            if semantic_mean is not None:
                semantic_features = semantic_mean.to(device=device, dtype=dtype)
            else:
                # Fallback: create zero semantic features matching acoustic dims
                semantic_features = torch.zeros(
                    acoustic_features.shape[0], acoustic_features.shape[1],
                    self.config.semantic_vae_dim, device=device, dtype=dtype
                )

            # Create speech_masks from cache dimensions (all frames valid)
            batch_size = acoustic_features.shape[0]
            seq_len = acoustic_features.shape[1]
            speech_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        elif speech_tensors is not None:
            # Encode speech through tokenizers
            with torch.no_grad():
                # Acoustic encoding
                if speech_tensors.dim() == 2:
                    speech_tensors = speech_tensors.unsqueeze(1)

                acoustic_output = self.acoustic_tokenizer.encode(speech_tensors)
                acoustic_features, _ = self.acoustic_tokenizer.sampling(acoustic_output)

                # Semantic encoding
                semantic_output = self.semantic_tokenizer.encode(speech_tensors)
                semantic_features = semantic_output.mean

            # Create speech_masks if not provided (all frames valid)
            if speech_masks is None:
                batch_size = acoustic_features.shape[0]
                seq_len = acoustic_features.shape[1]
                speech_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            # Return dummy features
            vae_dim = self.config.acoustic_vae_dim
            acoustic_features = torch.zeros(1, 1, vae_dim, device=device, dtype=dtype)
            semantic_features = torch.zeros(
                1, 1, self.config.semantic_vae_dim, device=device, dtype=dtype
            )
            speech_masks = torch.ones(1, 1, dtype=torch.bool, device=device)

        # Ensure acoustic and semantic have matching time dimensions
        acoustic_len = acoustic_features.shape[1]
        semantic_len = semantic_features.shape[1]
        if semantic_len < acoustic_len:
            pad_size = acoustic_len - semantic_len
            semantic_features = torch.nn.functional.pad(
                semantic_features, (0, 0, 0, pad_size), mode="constant", value=0
            )
        elif semantic_len > acoustic_len:
            semantic_features = semantic_features[:, :acoustic_len, :]

        # Apply scaling to acoustic features
        if not torch.isnan(self.speech_scaling_factor):
            acoustic_features = (
                acoustic_features + self.speech_bias_factor
            ) * self.speech_scaling_factor

        # Get embeddings through connectors
        acoustic_embed = self.acoustic_connector(acoustic_features)
        semantic_embed = self.semantic_connector(semantic_features)

        # Combine embeddings and index by speech_masks
        combined_embed = acoustic_embed + semantic_embed

        # Move speech_masks to CPU for indexing (matches working implementation)
        speech_embeds = combined_embed[speech_masks.cpu()]

        return acoustic_features, speech_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        voice_cache: Optional[dict] = None,
        logits_to_keep: Union[int, slice] = 0,
        **kwargs,
    ) -> Union[Tuple, KugelAudioCausalLMOutputWithPast]:
        """Forward pass for the model."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Process speech inputs if provided (raw audio or pre-encoded)
        if voice_cache is not None or (speech_tensors is not None and speech_masks is not None):
            _, speech_embeds = self._process_speech_inputs(
                speech_tensors.to(self.dtype) if speech_tensors is not None else None,
                speech_masks,
                voice_cache=voice_cache,
            )
            if speech_input_mask is not None:
                inputs_embeds[speech_input_mask] = speech_embeds

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return KugelAudioCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def sample_speech_tokens(
        self, condition: torch.Tensor, neg_condition: torch.Tensor, cfg_scale: float = 3.0
    ) -> torch.Tensor:
        """Sample speech latents using diffusion with classifier-free guidance."""
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)

        if cfg_scale == 1.0:
            # No CFG - single forward pass
            speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
            for t in self.model.noise_scheduler.timesteps:
                eps = self.model.prediction_head(
                    speech, t.repeat(speech.shape[0]).to(speech), condition=condition
                )
                speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
            return speech

        # With CFG - batched forward pass
        combined_condition = torch.cat([condition, neg_condition], dim=0).to(
            self.model.prediction_head.device
        )
        speech = torch.randn(combined_condition.shape[0], self.config.acoustic_vae_dim).to(
            combined_condition
        )

        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(
                combined, t.repeat(combined.shape[0]).to(combined), condition=combined_condition
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample

        return speech[: len(speech) // 2]

    @torch.no_grad()
    def generate(
        self,
        text_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        voice_prompt: Optional[torch.Tensor] = None,
        voice_cache: Optional[dict] = None,
        speech_tensors: Optional[torch.Tensor] = None,
        speech_masks: Optional[torch.Tensor] = None,
        speech_input_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 3.0,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        temperature: float = 1.0,
        show_progress: bool = True,
        apply_watermark: bool = True,
        **kwargs,
    ) -> KugelAudioGenerationOutput:
        """Generate speech from text.

        Supports voice cloning via:
        1. speech_tensors/speech_masks: Raw audio passed through processor
        2. voice_cache: Pre-encoded voice features (from encode_voice_prompt)
        3. voice_prompt: Legacy raw audio tensor (converted to speech_tensors)

        Args:
            text_ids: Tokenized text input (from processor)
            input_ids: Alternative name for text_ids
            voice_prompt: Voice audio tensor for cloning (legacy, use speech_tensors)
            voice_cache: Pre-encoded voice features
            speech_tensors: Voice audio tensor from processor
            speech_masks: Mask indicating valid voice frames
            speech_input_mask: Boolean mask indicating where to insert voice embeddings
            cfg_scale: Classifier-free guidance scale
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            show_progress: Whether to show progress bar
            apply_watermark: Whether to apply watermark

        Returns:
            KugelAudioGenerationOutput with sequences and speech_outputs
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Handle input_ids vs text_ids
        if text_ids is None and input_ids is not None:
            text_ids = input_ids
        if text_ids is None:
            raise ValueError("text_ids or input_ids is required")

        text_ids = text_ids.to(device)
        batch_size = text_ids.shape[0]

        # Handle legacy voice_prompt parameter
        if voice_prompt is not None and speech_tensors is None:
            speech_tensors = voice_prompt
            if speech_masks is None:
                audio_len = voice_prompt.shape[-1]
                num_frames = (audio_len + 3199) // 3200
                speech_masks = torch.ones(batch_size, num_frames, dtype=torch.bool, device=device)

        # Get special token IDs
        speech_start_id = getattr(self.config, "speech_start_id", None) or 151652
        speech_end_id = getattr(self.config, "speech_end_id", None) or 151653
        speech_diffusion_id = getattr(self.config, "speech_diffusion_id", None) or 151654
        eos_token_id = getattr(self.config.decoder_config, "eos_token_id", None) or 151643

        # Initialize streaming caches for tokenizers
        acoustic_cache_streaming = KugelAudioTokenizerStreamingCache()
        semantic_cache = KugelAudioTokenizerStreamingCache()

        # Initialize sequences and attention masks
        current_ids = text_ids
        attention_mask = torch.ones_like(current_ids)

        # For CFG, create negative prompt (just speech_start token)
        negative_ids = torch.full((batch_size, 1), speech_start_id, dtype=torch.long, device=device)
        negative_attention_mask = torch.ones_like(negative_ids)

        # Storage for generated audio and tracking
        audio_chunks = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Get initial embeddings
        inputs_embeds = self.model.get_input_embeddings()(current_ids)

        # Process voice/speech input if provided
        if speech_tensors is not None or voice_cache is not None:
            if voice_cache is not None:
                _, speech_embeds = self._process_speech_inputs(
                    speech_tensors=None,
                    speech_masks=None,
                    voice_cache=voice_cache,
                )
            else:
                speech_tensors = speech_tensors.to(device=device, dtype=dtype)
                if speech_masks is not None:
                    speech_masks = speech_masks.to(device)
                _, speech_embeds = self._process_speech_inputs(
                    speech_tensors=speech_tensors,
                    speech_masks=speech_masks,
                    voice_cache=None,
                )

            # Insert speech embeddings at positions marked by speech_input_mask
            if speech_input_mask is not None:
                speech_input_mask = speech_input_mask.to(device)
                inputs_embeds[speech_input_mask] = speech_embeds

        negative_inputs_embeds = self.model.get_input_embeddings()(negative_ids)

        # Setup logits processor to constrain to valid tokens
        valid_tokens = [speech_start_id, speech_end_id, speech_diffusion_id, eos_token_id]
        token_constraint = KugelAudioTokenConstraintProcessor(valid_tokens, device=device)

        # Initialize KV caches
        past_key_values = None
        negative_past_key_values = None

        # Progress bar
        progress_iter = (
            tqdm(range(max_new_tokens), desc="Generating", leave=False)
            if show_progress
            else range(max_new_tokens)
        )

        for step in progress_iter:
            if finished.all():
                break

            # Forward pass for positive (main) model
            if past_key_values is None:
                outputs = self(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                outputs = self(
                    inputs_embeds=inputs_embeds[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Apply token constraint
            logits = token_constraint(current_ids, logits)

            # Sample or greedy decode
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            # Force finished samples to output EOS
            next_tokens = torch.where(
                finished, torch.tensor(eos_token_id, device=device), next_tokens
            )

            # Update sequences
            current_ids = torch.cat([current_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype),
                ],
                dim=-1,
            )

            # Check for EOS tokens
            eos_mask = (next_tokens == eos_token_id) & ~finished
            if eos_mask.any():
                finished = finished | eos_mask

            # Check for speech_end tokens - mark as finished and clear caches
            speech_end_mask = (next_tokens == speech_end_id) & ~finished
            if speech_end_mask.any():
                finished = finished | speech_end_mask
                speech_end_indices = speech_end_mask.nonzero(as_tuple=False).squeeze(-1)
                acoustic_cache_streaming.set_to_zero(speech_end_indices)
                semantic_cache.set_to_zero(speech_end_indices)

            # Handle speech_start tokens - refresh negative model KV cache
            speech_start_mask = (next_tokens == speech_start_id) & ~finished
            if (
                speech_start_mask.any()
                and cfg_scale != 1.0
                and negative_past_key_values is not None
            ):
                speech_start_indices = speech_start_mask.nonzero(as_tuple=False).squeeze(-1)
                if speech_start_indices.dim() == 0:
                    speech_start_indices = speech_start_indices.unsqueeze(0)

                for sample_idx in speech_start_indices.tolist():
                    negative_attention_mask[sample_idx, :] = 0
                    negative_attention_mask[sample_idx, -1] = 1

                    key_caches, value_caches = _get_cache_tensors(negative_past_key_values)
                    for k_cache, v_cache in zip(key_caches, value_caches):
                        k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                        v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()

                    negative_ids[sample_idx, -1] = speech_start_id

            # Prepare next input embeddings
            next_inputs_embeds = self.model.get_input_embeddings()(next_tokens).unsqueeze(1)

            # Handle diffusion tokens - generate speech
            diffusion_mask = (next_tokens == speech_diffusion_id) & ~finished
            if diffusion_mask.any():
                diffusion_indices = diffusion_mask.nonzero(as_tuple=False).squeeze(-1)
                if diffusion_indices.dim() == 0:
                    diffusion_indices = diffusion_indices.unsqueeze(0)

                # Run negative forward pass for CFG
                if cfg_scale != 1.0:
                    if negative_past_key_values is None:
                        neg_outputs = self(
                            inputs_embeds=negative_inputs_embeds,
                            attention_mask=negative_attention_mask,
                            use_cache=True,
                            return_dict=True,
                        )
                    else:
                        neg_outputs = self(
                            inputs_embeds=negative_inputs_embeds[:, -1:],
                            attention_mask=negative_attention_mask,
                            past_key_values=negative_past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                    negative_past_key_values = neg_outputs.past_key_values

                    # Handle non-diffusion samples KV cache correction
                    non_diffusion_mask = ~diffusion_mask & ~finished
                    if non_diffusion_mask.any():
                        non_diffusion_indices = non_diffusion_mask.nonzero(as_tuple=False).squeeze(
                            -1
                        )
                        if non_diffusion_indices.dim() == 0:
                            non_diffusion_indices = non_diffusion_indices.unsqueeze(0)

                        key_caches, value_caches = _get_cache_tensors(negative_past_key_values)
                        for sample_idx in non_diffusion_indices.tolist():
                            start_idx = correct_cnt[sample_idx].item()
                            seq_len = negative_attention_mask.shape[1]

                            if start_idx + 1 < seq_len - 1:
                                negative_attention_mask[sample_idx, start_idx + 1 :] = (
                                    negative_attention_mask[sample_idx, start_idx:-1].clone()
                                )
                            negative_attention_mask[sample_idx, start_idx] = 0

                            for k_cache, v_cache in zip(key_caches, value_caches):
                                if start_idx + 1 < k_cache.shape[2] - 1:
                                    k_cache[sample_idx, :, start_idx + 1 :, :] = k_cache[
                                        sample_idx, :, start_idx:-1, :
                                    ].clone()
                                    v_cache[sample_idx, :, start_idx + 1 :, :] = v_cache[
                                        sample_idx, :, start_idx:-1, :
                                    ].clone()

                            if start_idx + 1 < negative_ids.shape[1] - 1:
                                negative_ids[sample_idx, start_idx + 1 :] = negative_ids[
                                    sample_idx, start_idx:-1
                                ].clone()

                        correct_cnt[non_diffusion_indices] += 1

                    neg_condition = neg_outputs.last_hidden_state[diffusion_indices, -1, :]
                else:
                    neg_condition = torch.zeros(
                        diffusion_indices.shape[0],
                        self.config.decoder_config.hidden_size,
                        device=device,
                        dtype=dtype,
                    )

                # Get conditioning from last hidden state
                condition = outputs.last_hidden_state[diffusion_indices, -1, :]

                # Sample speech latents using diffusion
                speech_latents = self.sample_speech_tokens(condition, neg_condition, cfg_scale)

                # Unscale latents
                scaled_latent = (
                    speech_latents / self.speech_scaling_factor - self.speech_bias_factor
                )

                # Decode through acoustic tokenizer with streaming cache
                audio = self.acoustic_tokenizer.decode(
                    scaled_latent.unsqueeze(1).permute(0, 2, 1),
                    cache=acoustic_cache_streaming,
                    sample_indices=diffusion_indices,
                    use_cache=True,
                )

                # Store audio chunks
                for i, idx in enumerate(diffusion_indices.tolist()):
                    if not finished[idx]:
                        audio_chunks[idx].append(audio[i].cpu())

                # Encode audio to semantic features with streaming cache
                semantic_output = self.semantic_tokenizer.encode(
                    audio,
                    cache=semantic_cache,
                    sample_indices=diffusion_indices,
                    use_cache=True,
                )
                semantic_features = semantic_output.mean

                # Compute embeddings for next step
                acoustic_embed = self.acoustic_connector(speech_latents.unsqueeze(1))
                semantic_embed = self.semantic_connector(semantic_features)
                diffusion_embeds = (acoustic_embed + semantic_embed).squeeze(1)

                # Update embeddings for diffusion samples
                next_inputs_embeds[diffusion_indices] = diffusion_embeds.unsqueeze(1)

            # Update embeddings for next iteration
            inputs_embeds = torch.cat([inputs_embeds, next_inputs_embeds], dim=1)

            # Update negative model
            negative_inputs_embeds = torch.cat([negative_inputs_embeds, next_inputs_embeds], dim=1)
            negative_attention_mask = torch.cat(
                [
                    negative_attention_mask,
                    torch.ones((batch_size, 1), device=device, dtype=negative_attention_mask.dtype),
                ],
                dim=-1,
            )
            negative_ids = torch.cat([negative_ids, next_tokens.unsqueeze(-1)], dim=-1)

        # Concatenate audio chunks with normalization
        speech_outputs = []
        for chunks in audio_chunks:
            if chunks:
                concatenated = torch.cat(chunks, dim=-1).squeeze()
                # Normalize audio to prevent clipping
                max_val = concatenated.abs().max()
                if max_val > 1.0:
                    concatenated = concatenated * (0.95 / max_val)
                # Apply watermark if enabled
                if apply_watermark:
                    concatenated = self._apply_watermark(concatenated, sample_rate=24000)
                speech_outputs.append(concatenated)
            else:
                speech_outputs.append(None)

        return KugelAudioGenerationOutput(
            sequences=current_ids,
            speech_outputs=speech_outputs,
        )

    @torch.no_grad()
    def encode_voice_prompt(
        self,
        voice_audio: torch.Tensor,
        sample_rate: int = 24000,
    ) -> dict:
        """Pre-encode a voice prompt for caching.

        Args:
            voice_audio: Raw audio tensor (1D, 2D [channels, samples], or 3D [batch, channels, samples])
            sample_rate: Sample rate of the audio (default 24000)

        Returns:
            Dict with acoustic_mean, acoustic_std, semantic_mean suitable for voice_cache
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if voice_audio.dim() == 1:
            voice_audio = voice_audio.unsqueeze(0).unsqueeze(0)
        elif voice_audio.dim() == 2:
            voice_audio = voice_audio.unsqueeze(1)

        voice_audio = voice_audio.to(device=device, dtype=dtype)

        with torch.no_grad():
            acoustic_output = self.model.acoustic_tokenizer.encode(voice_audio)
            semantic_output = self.model.semantic_tokenizer.encode(voice_audio)

        return {
            "acoustic_mean": acoustic_output.mean.cpu(),
            "acoustic_std": getattr(acoustic_output, "std", self.model.acoustic_tokenizer.fix_std),
            "semantic_mean": semantic_output.mean.cpu(),
            "audio_length": voice_audio.shape[-1],
            "sample_rate": sample_rate,
        }

    @staticmethod
    def load_voice(voice_path: str) -> dict:
        """Load a pre-encoded voice from a .pt file.

        Args:
            voice_path: Path to the .pt file containing pre-encoded voice features.
                The file should contain a dict with at least "acoustic_mean" tensor.

        Returns:
            Dict suitable for passing as voice_cache to generate().
        """
        voice_cache = torch.load(voice_path, map_location="cpu", weights_only=True)
        if "acoustic_mean" not in voice_cache:
            raise ValueError(
                f"Voice file {voice_path} does not contain 'acoustic_mean'. "
                f"Available keys: {list(voice_cache.keys())}"
            )
        return voice_cache

    def _apply_watermark(self, audio: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
        """Apply imperceptible watermark to generated audio.

        This watermark identifies audio as generated by KugelAudio and is designed
        to be robust against various audio transformations while remaining inaudible.
        """
        try:
            import torchaudio.functional as F
            from audioseal import AudioSeal
        except ImportError:
            return audio  # Graceful fallback if audioseal not available

        device = audio.device
        dtype = audio.dtype
        original_shape = audio.shape

        # Prepare audio for watermarking (AudioSeal expects [batch, channels, samples] at 16kHz)
        if audio.dim() == 1:
            audio_for_wm = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio_for_wm = audio.unsqueeze(0)
        else:
            audio_for_wm = audio

        audio_for_wm = audio_for_wm.float()

        # Resample to 16kHz for AudioSeal
        if sample_rate != 16000:
            audio_16k = F.resample(audio_for_wm, sample_rate, 16000)
        else:
            audio_16k = audio_for_wm

        # Load watermark generator (cached after first use)
        if not hasattr(self, "_wm_generator"):
            self._wm_generator = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
            self._wm_generator.eval()

        # Generate and apply watermark
        with torch.no_grad():
            watermark_16k = self._wm_generator.get_watermark(audio_16k.to(device), 16000)

        # Resample watermark back to original sample rate
        if sample_rate != 16000:
            watermark = F.resample(watermark_16k, 16000, sample_rate)
            # Ensure same length
            if watermark.shape[-1] != audio_for_wm.shape[-1]:
                if watermark.shape[-1] > audio_for_wm.shape[-1]:
                    watermark = watermark[..., : audio_for_wm.shape[-1]]
                else:
                    watermark = torch.nn.functional.pad(
                        watermark, (0, audio_for_wm.shape[-1] - watermark.shape[-1])
                    )
        else:
            watermark = watermark_16k

        # Add watermark to audio
        watermarked = audio_for_wm + watermark.to(audio_for_wm.device)

        # Normalize to prevent clipping
        max_val = watermarked.abs().max()
        if max_val > 1.0:
            watermarked = watermarked * (0.95 / max_val)

        # Restore original shape
        if len(original_shape) == 1:
            watermarked = watermarked.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            watermarked = watermarked.squeeze(0)

        return watermarked.to(dtype=dtype)


# Register with AutoModel
AutoModel.register(KugelAudioConfig, KugelAudioModel)
AutoModelForCausalLM.register(KugelAudioConfig, KugelAudioForConditionalGenerationInference)


__all__ = [
    "KugelAudioForConditionalGenerationInference",
    "KugelAudioCausalLMOutputWithPast",
    "KugelAudioGenerationOutput",
]
