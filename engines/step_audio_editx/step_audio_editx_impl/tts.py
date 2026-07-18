import hashlib
import os
import sys
import re
import logging
import numpy as np
import torch
import time
from typing import Tuple, Optional
from http import HTTPStatus

import torchaudio
# TTS Audio Suite patch: route audio loading through the shared librosa fallback layer.
from utils.audio.librosa_fallback import safe_load

# Add step_audio_editx_impl to sys.path so internal modules can import each other
_impl_dir = os.path.dirname(os.path.abspath(__file__))
if _impl_dir not in sys.path:
    sys.path.insert(0, _impl_dir)

from model_loader import model_loader, ModelSource
from transformers_compat import (
    STEP_AUDIO_TOKEN_END,
    STEP_AUDIO_TOKEN_START,
    STEP_VQ02_TOKEN_END,
    STEP_VQ06_TOKEN_START,
)
from .config.prompts import AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL, AUDIO_EDIT_SYSTEM_PROMPT
from stepvocoder.cosyvoice2.cli.cosyvoice import CosyVoice
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria

# Configure logging
logger = logging.getLogger(__name__)


class HTTPException(Exception):
    """Custom HTTP exception for API errors"""
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class InterruptionStoppingCriteria(StoppingCriteria):
    """
    StoppingCriteria that updates ComfyUI progress bar during generation.
    Matches reference implementation pattern.
    """

    def __init__(self, progress_bar, max_tokens):
        self.progress_bar = progress_bar
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.input_length = 0
        self.start_time = None
        self.last_print_time = None
        self.print_interval = 0.5  # Print progress every 0.5 seconds

    def _make_progress_bar(self, current, total, width=12):
        """Create ASCII progress bar: [████░░░░░░░░] current/total"""
        filled = int(width * current / total) if total > 0 else 0
        empty = width - filled
        bar = '█' * filled + '░' * empty
        return f"[{bar}] {current}/{total}"

    def __call__(self, input_ids, scores, **kwargs):
        """Called after each token generation to update progress."""
        # Store input length and start time on first call
        if self.input_length == 0:
            self.input_length = input_ids.shape[1]
            self.start_time = time.time()
            self.last_print_time = self.start_time
            print(f"\n[StepAudio] 🚀 Generation started (max {self.max_tokens} tokens)...")

        # Update progress
        new_tokens = input_ids.shape[1] - self.input_length
        if new_tokens > self.current_tokens:
            # Update ComfyUI progress bar with delta
            if self.progress_bar:
                self.progress_bar.update(new_tokens - self.current_tokens)
            self.current_tokens = new_tokens

            # Print progress with ASCII progress bar
            current_time = time.time()
            if current_time - self.last_print_time >= self.print_interval:
                block_elapsed = current_time - self.start_time
                it_per_sec = new_tokens / block_elapsed if block_elapsed > 0 else 0
                progress_bar_str = self._make_progress_bar(new_tokens, self.max_tokens)

                # Check if progress bar has job time estimation
                job_remaining = None
                job_elapsed = None
                if self.progress_bar:
                    if hasattr(self.progress_bar, 'get_job_remaining_str'):
                        job_remaining = self.progress_bar.get_job_remaining_str()
                    if hasattr(self.progress_bar, 'get_job_elapsed'):
                        job_elapsed = self.progress_bar.get_job_elapsed()

                # Use job elapsed if available, otherwise block elapsed
                elapsed = job_elapsed if job_elapsed else block_elapsed

                if job_remaining:
                    print(f"   Progress: {progress_bar_str} | {it_per_sec:.1f} it/s | {elapsed:.0f}s | {job_remaining}      ", end='\r')
                else:
                    print(f"   Progress: {progress_bar_str} | {it_per_sec:.1f} it/s | {elapsed:.0f}s      ", end='\r')
                self.last_print_time = current_time

        return False  # Never stop generation (let max_new_tokens handle it)


class RepetitionAwareLogitsProcessor(LogitsProcessor):
    """Apply the repetition guard used by the released PyTorch backend."""

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        window_size = 10
        threshold = 0.1
        window = input_ids[:, -window_size:]
        if window.shape[1] < window_size:
            return scores

        last_tokens = window[:, -1].unsqueeze(-1)
        repeat_counts = (window == last_tokens).sum(dim=1)
        repeat_ratios = repeat_counts.float() / window_size
        mask = repeat_ratios > threshold
        scores[mask, last_tokens[mask].squeeze(-1)] = float("-inf")
        return scores


class StepAudioTTS:
    """
    Step Audio TTS wrapper for voice cloning and audio editing tasks
    """

    def __init__(
        self,
        model_path,
        audio_tokenizer,
        model_source=ModelSource.AUTO,
        tts_model_id=None,
        quantization_config=None,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ):
        """
        Initialize StepAudioTTS

        Args:
            model_path: Model path
            audio_tokenizer: Audio tokenizer for wav2token processing
            model_source: Model source (auto/local/modelscope/huggingface)
            tts_model_id: TTS model ID, if None use model_path
            quantization_config: Quantization configuration ('int4', 'int8', or None)
            torch_dtype: PyTorch data type for model weights (default: torch.bfloat16)
            device_map: Device mapping for model (default: "cuda")
        """
        # Determine model ID or path to load
        if tts_model_id is None:
            tts_model_id = model_path

        # Configuration logged at debug level only
        logger.debug(f"StepAudioTTS config: source={model_source}, path={model_path}")

        self.audio_tokenizer = audio_tokenizer

        # Load LLM and tokenizer using model_loader
        try:
            self.llm, self.tokenizer, model_path = model_loader.load_transformers_model(
                tts_model_id,
                source=model_source,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device_map
            )

            # Model config is properly set via config.json (tie_word_embeddings=false)
            # No runtime patches needed - transformers versions 4.51.3+ handle this correctly
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

        # Load CosyVoice model (usually local path)
        self.cosy_model = CosyVoice(
            os.path.join(model_path, "CosyVoice-300M-25Hz")
        )

        # Use system prompts from config module
        self.edit_clone_sys_prompt_tpl = AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL
        self.edit_sys_prompt = AUDIO_EDIT_SYSTEM_PROMPT

    @torch.inference_mode()
    def clone(
        self,
        prompt_wav_path: str,
        prompt_text: str,
        target_text: str,
        progress_bar=None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """
        Clone voice from reference audio

        Args:
            prompt_wav_path: Path to reference audio file
            prompt_text: Text content of reference audio
            target_text: Text to synthesize with cloned voice
            progress_bar: ComfyUI progress bar for generation tracking
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Tuple[torch.Tensor, int]: Generated audio tensor and sample rate
        """
        # CRITICAL: Disable gradient computation for inference
        with torch.no_grad():
            try:
                logger.debug(f"Starting voice cloning: {prompt_wav_path}")
                prompt_wav, _ = torchaudio.load(prompt_wav_path)
                vq0206_codes, vq02_codes_ori, vq06_codes_ori, speech_feat, _, speech_embedding = (
                    self.preprocess_prompt_wav(prompt_wav_path)
                )
                prompt_speaker = self.generate_clone_voice_id(prompt_text, prompt_wav)

                # Use string tokens like original implementation
                # merge_vq0206_to_token_str uses original codes (vq02_codes_ori, vq06_codes_ori)
                prompt_wav_tokens = self.audio_tokenizer.merge_vq0206_to_token_str(
                    vq02_codes_ori, vq06_codes_ori
                )
                token_ids = self._encode_audio_edit_clone_prompt(
                    target_text,
                    prompt_text,
                    prompt_speaker,
                    prompt_wav_tokens,
                )

                print(f"🔄 Generating audio tokens for text: '{target_text[:50]}...' (input tokens: {len(token_ids)})")
                output_ids = self._generate_audio_tokens(
                    token_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    progress_bar=progress_bar,
                )

                logger.debug("Voice cloning generation completed")
                vq0206_codes_vocoder = torch.tensor([vq0206_codes], dtype=torch.long) - 65536
                return (
                    self.cosy_model.token2wav_nonstream(
                        output_ids - 65536,
                        vq0206_codes_vocoder,
                        speech_feat.to(torch.bfloat16),
                        speech_embedding.to(torch.bfloat16),
                    ),
                    24000,
                )
            except Exception as e:
                logger.error(f"Clone failed: {e}")
                raise

    @torch.inference_mode()
    def edit(
        self,
        input_audio_path: str,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None,
        progress_bar=None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """
        Edit audio based on specified edit type

        Args:
            input_audio_path: Path to input audio file
            audio_text: Text content of input audio
            edit_type: Type of edit (emotion, style, speed, etc.)
            edit_info: Specific edit information (happy, sad, etc.)
            text: Target text for para-linguistic editing
            progress_bar: ComfyUI progress bar for generation tracking
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Tuple[torch.Tensor, int]: Edited audio tensor and sample rate
        """
        try:
            logger.debug(f"Starting audio editing: {edit_type} - {edit_info}")
            vq0206_codes, vq02_codes_ori, vq06_codes_ori, speech_feat, _, speech_embedding = (
                self.preprocess_prompt_wav(input_audio_path)
            )
            audio_tokens = self.audio_tokenizer.merge_vq0206_to_token_str(
                vq02_codes_ori, vq06_codes_ori
            )
            # Build instruction prefix based on edit type
            instruct_prefix = self._build_audio_edit_instruction(audio_text, edit_type, edit_info, text)

            # Encode the complete prompt to token sequence
            prompt_tokens = self._encode_audio_edit_prompt(
                self.edit_sys_prompt, instruct_prefix, audio_tokens
            )

            logger.debug(f"Edit instruction: {instruct_prefix}")
            logger.debug(f"Encoded prompt length: {len(prompt_tokens)}")

            output_ids = self._generate_audio_tokens(
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                progress_bar=progress_bar,
            )
            vq0206_codes_vocoder = torch.tensor([vq0206_codes], dtype=torch.long) - 65536
            logger.debug("Audio editing generation completed")
            return (
                self.cosy_model.token2wav_nonstream(
                    output_ids - 65536,
                    vq0206_codes_vocoder,
                    speech_feat.to(torch.bfloat16),
                    speech_embedding.to(torch.bfloat16),
                ),
                24000,
            )
        except Exception as e:
            logger.error(f"Edit failed: {e}")
            raise

    def _build_audio_edit_instruction(
        self,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None
        ) -> str:
        """
        Build audio editing instruction based on request

        Args:
            audio_text: Text content of input audio
            edit_type: Type of edit
            edit_info: Specific edit information
            text: Target text for editing

        Returns:
            str: Instruction prefix
        """

        audio_text = audio_text.strip() if audio_text else ""
        if edit_type in {"emotion", "speed"}:
            if edit_info == "remove":
                instruct_prefix = f"Remove any emotion in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix=f"Make the following audio more {edit_info}. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "style":
            if edit_info == "remove":
                instruct_prefix = f"Remove any speaking styles in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix = f"Make the following audio more {edit_info} style. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "denoise":
            instruct_prefix = f"Remove any noise from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all noise from the audio.\n"
        elif edit_type == "vad":
            instruct_prefix = f"Remove any silent portions from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all silence from the audio.\n"
        elif edit_type == "paralinguistic":
            instruct_prefix = f"Add some non-verbal sounds to make the audio more natural, the new text is : {text}\n  The text corresponding to the audio is: {audio_text}\n"
        else:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Unsupported edit_type: {edit_type}",
            )

        return instruct_prefix

    def _encode_audio_edit_prompt(
        self, sys_prompt: str, instruct_prefix: str, audio_token_str: str
    ) -> list[int]:
        """
        Encode audio edit prompt to token sequence

        Args:
            sys_prompt: System prompt
            instruct_prefix: Instruction prefix
            audio_token_str: Audio tokens as string

        Returns:
            list[int]: Encoded token sequence
        """
        audio_token_str = audio_token_str.strip()
        history = [1]
        sys_tokens = self.tokenizer.encode(f"system\n{sys_prompt}")
        history.extend([4] + sys_tokens + [3])
        qrole_toks = self.tokenizer.encode("human\n")
        arole_toks = self.tokenizer.encode("assistant\n")
        human_turn_toks = self.tokenizer.encode(
            f"{instruct_prefix}\n{audio_token_str}\n"
        )
        history.extend([4] + qrole_toks + human_turn_toks + [3] + [4] + arole_toks)
        return history
    
    def _encode_audio_edit_clone_prompt(
        self, text: str, prompt_text: str, prompt_speaker: str, prompt_wav_tokens: str
    ):
        prompt = self.edit_clone_sys_prompt_tpl.format(
            speaker=prompt_speaker,
            prompt_text=prompt_text,
            prompt_wav_tokens=prompt_wav_tokens
        )
        sys_tokens = self.tokenizer.encode(f"system\n{prompt}")
        history = [1]
        history.extend([4] + sys_tokens + [3])
        prefix_tokens = self.tokenizer.encode("\n")
        target_token_encode = self.tokenizer.encode("\n" + text)
        target_tokens = target_token_encode[len(prefix_tokens):]
        qrole_toks = self.tokenizer.encode("human\n")
        arole_toks = self.tokenizer.encode("assistant\n")
        history.extend(
            [4] + qrole_toks + target_tokens + [3] + [4] + arole_toks
        )
        return history


    def detect_instruction_name(self, text):
        instruction_name = ""
        match_group = re.match(r"^([（\(][^\(\)()]*[）\)]).*$", text, re.DOTALL)
        if match_group is not None:
            instruction = match_group.group(1)
            instruction_name = instruction.strip("()（）")
        return instruction_name

    def process_audio_file(self, audio_path: str) -> Tuple[any, int]:
        """
        Process audio file and return numpy array and sample rate

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple[numpy.ndarray, int]: Audio data and sample rate
        """
        try:
            # TTS Audio Suite patch: avoid raw librosa.load here because fragile numba/librosa
            # environments can fail before EditX even starts inference.
            audio_data, sample_rate = safe_load(audio_path, sr=22050, mono=True)
            logger.debug(f"Audio file processed successfully: {audio_path}")
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Failed to process audio file: {e}")
            raise

    def preprocess_prompt_wav(self, prompt_wav_path : str):
        prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)  # 将多通道音频转换为单通道

        # volume-normalize avoid clipping
        norm = torch.max(torch.abs(prompt_wav), dim=1, keepdim=True)[0]
        if norm > 0.6: # hard code;  max absolute value is 0.6
            prompt_wav = prompt_wav / norm * 0.6 

        speech_feat, speech_feat_len = self.cosy_model.frontend.extract_speech_feat(
            prompt_wav, prompt_wav_sr
        )
        speech_embedding = self.cosy_model.frontend.extract_spk_embedding(
            prompt_wav, prompt_wav_sr
        )
        vq0206_codes, vq02_codes_ori, vq06_codes_ori = self.audio_tokenizer.wav2token(prompt_wav, prompt_wav_sr)
        return (
            vq0206_codes,
            vq02_codes_ori,
            vq06_codes_ori,
            speech_feat,
            speech_feat_len,
            speech_embedding,
        )
        
    def _generate_audio_tokens(
        self,
        prompt_token_ids,
        *,
        max_new_tokens,
        temperature,
        do_sample,
        progress_bar,
    ):
        """Generate Step audio tokens with settings matching upstream vLLM."""
        device = next(self.llm.parameters()).device
        input_tensor = torch.tensor(
            [prompt_token_ids],
            dtype=torch.long,
            device=device,
        )
        stopping_criteria = None
        if progress_bar is not None:
            from transformers.generation.stopping_criteria import StoppingCriteriaList

            stopping_criteria = StoppingCriteriaList([
                InterruptionStoppingCriteria(progress_bar, max_new_tokens)
            ])

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "stopping_criteria": stopping_criteria,
            "logits_processor": LogitsProcessorList([
                RepetitionAwareLogitsProcessor()
            ]),
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self.llm.generate(input_tensor, **generation_kwargs)
        return self._extract_generated_audio_tokens(
            output_ids,
            len(prompt_token_ids),
        )

    def generate_clone_voice_id(self, prompt_text, prompt_wav):
        hasher = hashlib.sha256()
        hasher.update(prompt_text.encode("utf-8"))
        wav_data = prompt_wav.cpu().numpy()
        if wav_data.size > 2000:
            audio_sample = np.concatenate(
                [wav_data.flatten()[:1000], wav_data.flatten()[-1000:]]
            )
        else:
            audio_sample = wav_data.flatten()
        hasher.update(audio_sample.tobytes())
        return f"clone_{hasher.hexdigest()[:16]}"

    def _get_eos_token_id(self):
        eos_token_id = getattr(self.llm.generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.llm.config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            return eos_token_id[0] if eos_token_id else None
        return eos_token_id

    def _extract_generated_audio_tokens(self, output_ids, prompt_length):
        """Return generated Step audio tokens, excluding the prompt and EOS."""
        # TTS Audio Suite patch: never pass EOS/text tokens to the vocoder. Newer
        # Transformers versions can otherwise turn a failed generation into 0s audio.
        generated = output_ids[:, prompt_length:]
        eos_token_id = self._get_eos_token_id()
        if eos_token_id is not None and generated.shape[1] > 0:
            eos_positions = generated[0].eq(eos_token_id)
            if eos_positions.any():
                first_eos = int(torch.nonzero(eos_positions, as_tuple=False)[0].item())
                generated = generated[:, :first_eos]

        # Step emits groups of two 1,024-entry VQ02 codes followed by three
        # 4,096-entry VQ06 codes. VQ06 is offset by 1,024 in the mixed stream.
        positions = torch.arange(generated.shape[1], device=generated.device)
        vq02_slots = (positions % 5) < 2
        valid_vq02 = (
            (generated >= STEP_AUDIO_TOKEN_START)
            & (generated < STEP_VQ02_TOKEN_END)
        )
        valid_vq06 = (
            (generated >= STEP_VQ06_TOKEN_START)
            & (generated < STEP_AUDIO_TOKEN_END)
        )
        valid_audio = torch.where(
            vq02_slots.unsqueeze(0),
            valid_vq02,
            valid_vq06,
        )
        if generated.shape[1] < 5 or not bool(valid_audio.all().item()):
            invalid_count = int((~valid_audio).sum().item()) if generated.numel() else 0
            flat_tokens = generated[0].detach().cpu().tolist()
            runs = []
            if flat_tokens:
                run_start = 0
                run_is_audio = bool(valid_audio[0, 0].item())
                for position, token_id in enumerate(flat_tokens[1:], start=1):
                    is_audio = bool(valid_audio[0, position].item())
                    if is_audio != run_is_audio:
                        runs.append(
                            f"{'audio' if run_is_audio else 'other'}:"
                            f"{run_start}-{position - 1}"
                        )
                        run_start = position
                        run_is_audio = is_audio
                runs.append(
                    f"{'audio' if run_is_audio else 'other'}:"
                    f"{run_start}-{len(flat_tokens) - 1}"
                )

            invalid_examples = []
            for position, token_id in enumerate(flat_tokens):
                if bool(valid_audio[0, position].item()):
                    continue
                token_text = self.tokenizer.convert_ids_to_tokens(token_id)
                invalid_examples.append(f"{position}:{token_id}={token_text!r}")
                if len(invalid_examples) == 12:
                    break

            print(
                "🔬 Step output token runs: "
                + (
                    ", ".join(runs[:24])
                    + (f", … ({len(runs)} runs total)" if len(runs) > 24 else "")
                    if runs
                    else "(empty)"
                )
            )
            print(
                "🔬 First non-audio tokens: "
                + (", ".join(invalid_examples) if invalid_examples else "(none)")
            )
            raise RuntimeError(
                "Step Audio EditX generated no usable speech tokens "
                f"({generated.shape[1]} tokens, {invalid_count} outside the audio vocabulary)."
            )
        return generated
