"""High-level generation utilities for KugelAudio."""

from typing import Optional, Union

import torch


def load_model_and_processor(
    model_name_or_path: str = "kugelaudio/kugelaudio-0-open",
    device: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_flash_attention: bool = True,
):
    """Load KugelAudio model and processor.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        device: Device to load model on (auto-detected if None)
        torch_dtype: Data type for model weights
        use_flash_attention: Whether to use flash attention if available

    Returns:
        Tuple of (model, processor)

    Example:
        >>> model, processor = load_model_and_processor("kugelaudio/kugelaudio-0-open")
    """
    from ..models import KugelAudioForConditionalGenerationInference
    from ..processors import KugelAudioProcessor

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect dtype
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load model
    attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"
    try:
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        ).to(device)
    except Exception:
        # Fallback without flash attention
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(device)

    model.eval()

    # Strip encoders to save VRAM (only decoders needed for inference)
    model.model.strip_encoders()

    # Load processor
    processor = KugelAudioProcessor.from_pretrained(model_name_or_path)

    return model, processor


def generate_speech(
    model,
    processor,
    text: str,
    voice: Optional[str] = None,
    cfg_scale: float = 3.0,
    max_new_tokens: int = 4096,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Generate speech from text using an optional pre-encoded voice.

    Voice cloning from raw audio is no longer supported. Use a pre-encoded
    voice name from the voices.json registry instead.

    All generated audio is automatically watermarked for identification.

    Args:
        model: KugelAudio model
        processor: KugelAudio processor
        text: Text to synthesize
        voice: Name of a pre-encoded voice (from voices.json registry)
        cfg_scale: Classifier-free guidance scale
        max_new_tokens: Maximum number of tokens to generate
        device: Device for generation

    Returns:
        Generated audio tensor (watermarked)

    Example:
        >>> audio = generate_speech(model, processor, "Hello world!", voice="default")
        >>> processor.save_audio(audio, "output.wav")
    """
    if device is None:
        device = next(model.parameters()).device

    # Process inputs with optional pre-encoded voice
    inputs = processor(text=text, voice=voice, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate (watermark is automatically applied by the model)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            cfg_scale=cfg_scale,
            max_new_tokens=max_new_tokens,
        )

    audio = outputs.speech_outputs[0] if outputs.speech_outputs else None

    if audio is None:
        raise RuntimeError("Generation failed - no audio output")

    return audio
