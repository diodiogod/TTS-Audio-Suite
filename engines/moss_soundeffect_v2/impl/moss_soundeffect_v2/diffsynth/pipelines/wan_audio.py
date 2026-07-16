import torch, warnings, glob, os, types, math, json
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union, Literal
from dataclasses import dataclass
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from safetensors.torch import load_file

from ..utils import BasePipeline, PipelineUnit, PipelineUnitRunner
from ..models.utils import load_state_dict
from ..models.wan_audio_dit import WanAudioModel
from ..models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from ..models.qwen3_text_encoder import Qwen3TextEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter

from diffusers import AutoencoderOobleck
from ..models.dac_vae import DAC


_WAN_AUDIO_DIT_PRESETS: dict[str, dict[str, int]] = {
    # Only the key DiT hyperparameters are listed here. Other settings
    # (text_dim / freq_dim / patch_size, ...) are kept constant.
    "1.3B": {"dim": 1536, "ffn_dim": 8960, "num_heads": 12, "num_layers": 30},
    "5B": {"dim": 3072, "ffn_dim": 14336, "num_heads": 24, "num_layers": 30},
    "500M": {"dim": 1024, "ffn_dim": 5120, "num_heads": 8, "num_layers": 28},
    "300M": {"dim": 896, "ffn_dim": 4096, "num_heads": 7, "num_layers": 20},
    "1B": {"dim": 1408, "ffn_dim": 7168, "num_heads": 11, "num_layers": 28},
    "100M": {"dim": 512, "ffn_dim": 2560, "num_heads": 4, "num_layers": 16}
}


def _resolve_wan_audio_dit_preset(dit_variant: str):
    # Accept lowercase / shorthand inputs (e.g. "5b", "1.3b").
    v = (dit_variant or "1.3B").strip()
    v_norm = {"1.3b": "1.3B", "5b": "5B", "500m": "500M", "300m": "300M", "1b": "1B", "100m": "100M"}.get(v.lower(), v)
    if v_norm not in _WAN_AUDIO_DIT_PRESETS:
        raise ValueError(
            f"Invalid dit_variant: {dit_variant!r}. Supported: {sorted(_WAN_AUDIO_DIT_PRESETS.keys())}"
        )
    return _WAN_AUDIO_DIT_PRESETS[v_norm], v_norm


# Maps diffusers-style keys in the exported HF DiT checkpoint back to the
# native WanAudioModel keys. Paired with the forward direction in
# moss_soundeffect_v2/hf_export.py.
_HF_DIT_BLOCK_RENAME = {
    "attn1.norm_k.weight": "self_attn.norm_k.weight",
    "attn1.norm_q.weight": "self_attn.norm_q.weight",
    "attn1.to_k.bias": "self_attn.k.bias",
    "attn1.to_k.weight": "self_attn.k.weight",
    "attn1.to_out.0.bias": "self_attn.o.bias",
    "attn1.to_out.0.weight": "self_attn.o.weight",
    "attn1.to_q.bias": "self_attn.q.bias",
    "attn1.to_q.weight": "self_attn.q.weight",
    "attn1.to_v.bias": "self_attn.v.bias",
    "attn1.to_v.weight": "self_attn.v.weight",
    "attn2.norm_k.weight": "cross_attn.norm_k.weight",
    "attn2.norm_q.weight": "cross_attn.norm_q.weight",
    "attn2.to_k.bias": "cross_attn.k.bias",
    "attn2.to_k.weight": "cross_attn.k.weight",
    "attn2.to_out.0.bias": "cross_attn.o.bias",
    "attn2.to_out.0.weight": "cross_attn.o.weight",
    "attn2.to_q.bias": "cross_attn.q.bias",
    "attn2.to_q.weight": "cross_attn.q.weight",
    "attn2.to_v.bias": "cross_attn.v.bias",
    "attn2.to_v.weight": "cross_attn.v.weight",
    "ffn.net.0.proj.bias": "ffn.0.bias",
    "ffn.net.0.proj.weight": "ffn.0.weight",
    "ffn.net.2.bias": "ffn.2.bias",
    "ffn.net.2.weight": "ffn.2.weight",
    "norm2.bias": "norm3.bias",
    "norm2.weight": "norm3.weight",
    "scale_shift_table": "modulation",
}

_HF_DIT_GLOBAL_RENAME = {
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "patch_embedding.weight": "patch_embedding.weight",
}


def _convert_hf_dit_state_dict(state_dict: dict) -> dict:
    out = {}
    for key, param in state_dict.items():
        if key in _HF_DIT_GLOBAL_RENAME:
            out[_HF_DIT_GLOBAL_RENAME[key]] = param
        elif key.startswith("blocks."):
            parts = key.split(".", 2)
            block_idx, suffix = parts[1], parts[2]
            if suffix in _HF_DIT_BLOCK_RENAME:
                out[f"blocks.{block_idx}.{_HF_DIT_BLOCK_RENAME[suffix]}"] = param
            else:
                out[key] = param
        else:
            out[key] = param
    return out


class WanAudioPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None, flow_shift=5.0):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=flow_shift, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder = None
        self.image_encoder = None
        self.dit: WanAudioModel = None
        self.dit2: WanAudioModel = None
        self.vae: AutoencoderOobleck | DAC = None
        self.motion_controller = None
        self.vace = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanAudioUnit_ShapeChecker(),
            WanAudioUnit_NoiseInitializer(),
            WanAudioUnit_InputAudioEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            # WanVideoUnit_ImageEmbedderVAE(),
            # WanVideoUnit_ImageEmbedderCLIP(),
            # WanVideoUnit_ImageEmbedderFused(),
            # WanVideoUnit_FunControl(),
            # WanVideoUnit_FunReference(),
            # WanVideoUnit_FunCameraControl(),
            # WanVideoUnit_SpeedControl(),
            # WanVideoUnit_VACE(),
            # WanVideoUnit_UnifiedSequenceParallel(),
            # WanVideoUnit_TeaCache(),
            # WanVideoUnit_CfgMerger(),
        ]
        self.model_fn = model_fn_wan_video

    def training_loss(self, **inputs):
        # num_train_timesteps is the size of the timestep candidate pool (1000),
        # not the number of optimizer steps per epoch.
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        bs = inputs["input_latents"].size(0)
        # One random timestep id per sample, uniform in [min, max).
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (bs,))
        timestep = self.scheduler.timesteps[timestep_id].to(device=self.device)

        with torch.autocast("cuda", dtype=torch.float32):
            inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
            # Flow-match target: predict (noise - x_0), i.e. the velocity vector.
            training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        # DiT forward (with grad): noisy latent (B, 128, T), timestep, text context.
        noise_pred = self.model_fn(**inputs, timestep=timestep)

        # MSE between the model's predicted velocity and the true velocity.
        # Classic DDPM targets epsilon (MSE(model_output, eps)); flow matching
        # targets (epsilon - x_0), so the same head is reinterpreted as a
        # vector-field predictor here.
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        return loss


    def check_resize_num_channels_num_samples(self, num_channels, num_samples):
        # Shape check
        if num_samples % self.num_samples_division_factor != 0:
            num_samples = num_samples // self.num_samples_division_factor * self.num_samples_division_factor
            # print(f"num_samples % {self.num_samples_division_factor} != 0. We round it down to {num_samples}.")
        return num_channels, num_samples


    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: Union[str, torch.device] = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "WanAudioPipeline":
        """Load a WanAudioPipeline from a HuggingFace-format directory.

        Expected layout (a diffusers-style HF model directory):
            model_dir/
                model_index.json
                scheduler/scheduler_config.json
                transformer/config.json
                transformer/diffusion_pytorch_model.safetensors
                text_encoder/...        (Qwen3)
                tokenizer/...
                vae/vae_128d_48k.pth    (or diffusion_pytorch_model.safetensors)
        """
        with open(os.path.join(model_dir, "model_index.json")) as f:
            index = json.load(f)
        print(f"Loading from: {model_dir}")
        print(f"  Pipeline: {index['_class_name']}, dit_variant: {index.get('dit_variant')}")

        with open(os.path.join(model_dir, "scheduler", "scheduler_config.json")) as f:
            sched_cfg = json.load(f)
        with open(os.path.join(model_dir, "transformer", "config.json")) as f:
            dit_cfg = json.load(f)

        te_path = os.path.join(model_dir, "text_encoder")
        print(f"  Loading text_encoder from {te_path} ...")
        text_encoder = Qwen3TextEncoder(te_path, torch_dtype=torch_dtype)
        text_encoder = text_encoder.to(device)
        print(f"  text_encoder: dim={text_encoder.dim}")

        tok_path = os.path.join(model_dir, "tokenizer")
        print(f"  Loading tokenizer from {tok_path} ...")
        prompter = WanPrompter(tokenizer_path=tok_path)
        prompter.fetch_models(text_encoder)

        vae_dir = os.path.join(model_dir, "vae")
        vae_pth = os.path.join(vae_dir, "vae_128d_48k.pth")
        vae_safetensors = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
        if os.path.exists(vae_pth):
            print(f"  Loading DAC VAE from {vae_pth} ...")
            vae = DAC.load(vae_pth)
        elif os.path.exists(vae_safetensors):
            print(f"  Loading DAC VAE from {vae_safetensors} ...")
            vae = DAC.load(vae_safetensors)
        else:
            raise FileNotFoundError(f"No VAE found in {vae_dir}")

        dit_weights_path = os.path.join(model_dir, "transformer", "diffusion_pytorch_model.safetensors")
        print(f"  Loading DiT from {dit_weights_path} ...")
        diffusers_sd = load_file(dit_weights_path)
        custom_sd = _convert_hf_dit_state_dict(diffusers_sd)

        dit = WanAudioModel(
            in_dim=dit_cfg["in_dim"],
            out_dim=dit_cfg["out_dim"],
            text_dim=dit_cfg["text_dim"],
            freq_dim=dit_cfg["freq_dim"],
            eps=dit_cfg["eps"],
            patch_size=tuple(dit_cfg["patch_size"]),
            has_image_input=dit_cfg["has_image_input"],
            dim=dit_cfg["dim"],
            ffn_dim=dit_cfg["ffn_dim"],
            num_heads=dit_cfg["num_heads"],
            num_layers=dit_cfg["num_layers"],
            vae_type=dit_cfg.get("vae_type", "dac"),
        )
        load_result = dit.load_state_dict(custom_sd)
        print(
            f"  DiT loaded: missing={len(load_result.missing_keys)}, "
            f"unexpected={len(load_result.unexpected_keys)}"
        )

        pipe = cls(
            device=device,
            torch_dtype=torch_dtype,
            flow_shift=sched_cfg.get("shift", 5.0),
        )
        pipe.text_encoder = text_encoder
        pipe.prompter = prompter
        pipe.vae = vae
        pipe.dit = dit
        pipe.audio_latent_dim = dit_cfg["in_dim"]
        pipe.num_samples_division_factor = vae.hop_length
        pipe.dit_variant = index.get("dit_variant")
        pipe.to(device)
        print(f"  Pipeline assembled on {device}")
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: Union[str, list[str]],
        negative_prompt: Optional[Union[str, list[str]]] = "",
        # Image-to-video
        # input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        # end_image: Optional[Image.Image] = None,
        # Video-to-video
        # input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # ControlNet
        # control_video: Optional[list[Image.Image]] = None,
        # reference_image: Optional[Image.Image] = None,
        # Camera control
        # camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        # camera_control_speed: Optional[float] = 1/54,
        # camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        # vace_video: Optional[list[Image.Image]] = None,
        # vace_video_mask: Optional[Image.Image] = None,
        # vace_reference_image: Optional[Image.Image] = None,
        # vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        # height: Optional[int] = 480,
        # width: Optional[int] = 832,
        # num_frames=81,
        num_samples=44100*10,
        num_channels=2,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        # motion_bucket_id: Optional[int] = None,
        # VAE tiling
        # tiled: Optional[bool] = True,
        # tile_size: Optional[tuple[int, int]] = (30, 52),
        # tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        # sliding_window_size: Optional[int] = None,
        # sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        # print(f"{self.scheduler.timesteps = }")
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        # Infer batch size; prompt may be a list[str].
        computed_batch_size = len(prompt) if isinstance(prompt, (list, tuple)) else 1
        # For batched input, broadcast a single negative_prompt to the same length.
        if computed_batch_size > 1 and not isinstance(negative_prompt, (list, tuple)):
            inputs_nega["negative_prompt"] = [negative_prompt] * computed_batch_size
        # inputs_shared = {
        #     "input_image": input_image,
        #     "end_image": end_image,
        #     "input_video": input_video, "denoising_strength": denoising_strength,
        #     "control_video": control_video, "reference_image": reference_image,
        #     "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
        #     "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
        #     "seed": seed, "rand_device": rand_device,
        #     "height": height, "width": width, "num_frames": num_frames,
        #     "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
        #     "sigma_shift": sigma_shift,
        #     "motion_bucket_id": motion_bucket_id,
        #     "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
        #     "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
        # }
        inputs_shared = {
            "num_samples": num_samples,
            "num_channels": num_channels,
            "denoising_strength": denoising_strength,
            "seed": seed, "rand_device": rand_device,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "batch_size": computed_batch_size,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(device=self.device)
            
            # Inference
            torch.compiler.cudagraph_mark_step_begin()
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_posi = noise_pred_posi.clone()
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred_posi = noise_pred_posi.float()
                noise_pred_nega = noise_pred_nega.float()
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # VACE (TODO: remove it)
        # if vace_reference_image is not None:
        #     inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"]
        max_decode_bs = 8
        audio_chunks = []
        for start in range(0, latents.size(0), max_decode_bs):
            end = min(start + max_decode_bs, latents.size(0))
            with torch.autocast("cuda", dtype=torch.float32):
                if isinstance(self.vae, DAC):
                    audio_chunk = self.vae.decode(latents[start:end])
                else:
                    audio_chunk = self.vae.decode(latents[start:end]).sample
            audio_chunks.append(audio_chunk)
        audio = torch.cat(audio_chunks, dim=0)
        # video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return audio



class WanAudioUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("num_channels", "num_samples"))

    def process(self, pipe: WanAudioPipeline, num_channels, num_samples):
        num_channels, num_samples = pipe.check_resize_num_channels_num_samples(num_channels, num_samples)
        return {"num_channels": num_channels, "num_samples": num_samples}



class WanAudioUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("input_audio", "num_samples", "seed", "rand_device", "batch_size"))

    def process(self, pipe: WanAudioPipeline, input_audio, num_samples, seed, rand_device, batch_size):
        if input_audio is not None:
            bsz = input_audio.size(0) if input_audio.ndim == 3 else 1
        else:
            bsz = batch_size if batch_size is not None else 1
        shape = (bsz, pipe.audio_latent_dim, num_samples // pipe.num_samples_division_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}
    


class WanAudioUnit_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_audio", "audio_latent", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanAudioPipeline, input_audio, audio_latent, noise, tiled, tile_size, tile_stride, vace_reference_image):
        # Pass-through branch when an audio_latent is provided directly.
        if audio_latent is not None:
            latents = audio_latent
            if latents.ndim == 2:
                latents = latents.unsqueeze(0)
            latents = latents.to(dtype=pipe.torch_dtype, device=pipe.device)
            if pipe.scheduler.training:
                return {"latents": noise, "input_latents": latents}
            else:
                latents = pipe.scheduler.add_noise(latents, noise, timestep=pipe.scheduler.timesteps[0])
                return {"latents": latents}

        if input_audio is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        if input_audio.ndim == 2:
            # add batch dim
            input_audio = input_audio.unsqueeze(0)
        # print(f"{input_audio.shape = }")
        # from time import perf_counter
        # start_time = perf_counter()
        with torch.autocast("cuda", dtype=torch.float32):
            if isinstance(pipe.vae, DAC):
                input_latents = pipe.vae.encode(input_audio)[0].mode()
            else:
                input_latents = pipe.vae.encode(input_audio).latent_dist.mode()
        input_latents = input_latents.to(device=pipe.device)
        # print(f"{input_latents.mean() = }, {input_latents.std() = }")
        # end_time = perf_counter()
        # print(f"vae.encode time taken: {end_time - start_time} seconds")
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanAudioPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        # from time import perf_counter
        # start_time = perf_counter()
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        # end_time = perf_counter()
        # print(f"prompter.encode_prompt time taken: {end_time - start_time} seconds")
        return {"context": prompt_emb}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanAudioPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanAudioPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanAudioPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanAudioModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



@torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def model_fn_wan_video(
    dit: WanAudioModel,
    motion_controller = None,
    vace = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        raise ValueError("not supported")
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        with torch.autocast("cuda", dtype=torch.float32):
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
            # print(f"{t.shape = }")
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
            # print(f"{t_mod.shape = }")
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        raise ValueError("not supported")
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Add camera control
    x, (f, ) = dit.patchify(x, control_camera_latents_input)
    # print(f"{f = }")
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    # freqs is now a registered buffer (moves with model.to(device)). Do not
    # write Python attributes here so torch.compile can trace through.
    audio_freqs = dit.freqs
    freqs = torch.cat([
        audio_freqs[0][:f].view(f, -1).expand(f, -1),
        audio_freqs[1][:f].view(f, -1).expand(f, -1),
        audio_freqs[2][:f].view(f, -1).expand(f, -1),
    ], dim=-1).reshape(f, 1, -1)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, ))
    return x
