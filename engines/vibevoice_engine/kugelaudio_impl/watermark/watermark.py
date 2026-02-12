"""Audio watermarking for KugelAudio using Facebook's AudioSeal.

AudioSeal provides state-of-the-art speech localized watermarking with:
- High robustness to audio editing and compression
- Fast single-pass detection (real-time capable)
- Sample-level detection (1/16k second resolution)
- Optional 16-bit message embedding

Reference: https://huggingface.co/facebook/audioseal
"""

from typing import Optional, Union, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import torch

# Try to import AudioSeal
try:
    from audioseal import AudioSeal
    AUDIOSEAL_AVAILABLE = True
except ImportError:
    AUDIOSEAL_AVAILABLE = False
    warnings.warn(
        "AudioSeal not installed. Install with: pip install audioseal\n"
        "Watermarking will use fallback implementation."
    )


@dataclass
class WatermarkResult:
    """Result of watermark detection."""
    detected: bool
    confidence: float
    message: Optional[torch.Tensor] = None
    frame_probabilities: Optional[torch.Tensor] = None


class AudioWatermark:
    """Professional audio watermarking using Facebook's AudioSeal.
    
    AudioSeal is a state-of-the-art watermarking system that embeds
    imperceptible watermarks in audio that are robust to various
    audio transformations.
    
    Features:
    - Imperceptible watermarks with minimal quality degradation
    - Robust to compression, resampling, and editing
    - Fast detection suitable for real-time applications
    - Optional 16-bit message embedding for tracking
    
    Example:
        >>> watermark = AudioWatermark()
        >>> watermarked_audio = watermark.embed(audio)
        >>> result = watermark.detect(watermarked_audio)
        >>> print(f"Detected: {result.detected}, Confidence: {result.confidence:.2%}")
    
    Args:
        model_name: AudioSeal model variant ("audioseal_wm_16bits")
        device: Device for inference ("cuda" or "cpu")
        message: Optional 16-bit message to embed (for tracking)
    """
    
    # Default message identifying KugelAudio-generated content
    KUGELAUDIO_MESSAGE = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0, 
                                        0, 1, 0, 1, 0, 1, 0, 1]])  # Alternating pattern
    
    # AudioSeal expects 16kHz audio
    AUDIOSEAL_SAMPLE_RATE = 16000
    
    def __init__(
        self,
        model_name: str = "audioseal_wm_16bits",
        detector_name: str = "audioseal_detector_16bits",
        device: Optional[Union[str, torch.device]] = None,
        message: Optional[torch.Tensor] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self._generator = None
        self._detector = None
        self._model_name = model_name
        self._detector_name = detector_name
        
        # Use KugelAudio identifier message by default
        self.message = message if message is not None else self.KUGELAUDIO_MESSAGE.clone()
        
        if not AUDIOSEAL_AVAILABLE:
            warnings.warn(
                "AudioSeal not available. Watermarking disabled. "
                "Install with: pip install audioseal"
            )
    
    @property
    def generator(self):
        """Lazy load the generator model."""
        if self._generator is None and AUDIOSEAL_AVAILABLE:
            self._generator = AudioSeal.load_generator(self._model_name)
            self._generator = self._generator.to(self.device)
            self._generator.eval()
        return self._generator
    
    @property
    def detector(self):
        """Lazy load the detector model."""
        if self._detector is None and AUDIOSEAL_AVAILABLE:
            self._detector = AudioSeal.load_detector(self._detector_name)
            self._detector = self._detector.to(self.device)
            self._detector.eval()
        return self._detector
    
    def _resample(
        self, 
        audio: torch.Tensor, 
        orig_sr: int, 
        target_sr: int
    ) -> torch.Tensor:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        try:
            import torchaudio.functional as F
            return F.resample(audio, orig_sr, target_sr)
        except ImportError:
            # Fallback using scipy
            from scipy import signal
            audio_np = audio.cpu().numpy()
            num_samples = int(len(audio_np.flatten()) * target_sr / orig_sr)
            resampled = signal.resample(audio_np.flatten(), num_samples)
            return torch.from_numpy(resampled).reshape(audio.shape[0], audio.shape[1], -1).to(audio.device)
    
    def embed(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 24000,
        message: Optional[torch.Tensor] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Embed watermark into audio.
        
        The watermark is imperceptible and robust to various audio
        transformations including compression and resampling.
        
        Args:
            audio: Input audio of shape (samples,), (channels, samples), 
                   or (batch, channels, samples)
            sample_rate: Sample rate of input audio (default: 24000 for KugelAudio)
            message: Optional 16-bit message to embed
            
        Returns:
            Watermarked audio with same shape and type as input
        """
        if not AUDIOSEAL_AVAILABLE:
            # Return unchanged if AudioSeal not available
            return audio
        
        # Track input type
        is_numpy = isinstance(audio, np.ndarray)
        if is_numpy:
            audio = torch.from_numpy(audio)
        
        original_device = audio.device
        original_dtype = audio.dtype
        
        # Ensure float32 for processing
        audio = audio.float()
        
        # Handle different input shapes
        original_shape = audio.shape
        if audio.ndim == 1:
            # (samples,) -> (1, 1, samples)
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.ndim == 2:
            # (channels, samples) -> (1, channels, samples)
            audio = audio.unsqueeze(0)
        
        # Move to device
        audio = audio.to(self.device)
        
        # Resample to 16kHz for AudioSeal
        if sample_rate != self.AUDIOSEAL_SAMPLE_RATE:
            audio_16k = self._resample(audio, sample_rate, self.AUDIOSEAL_SAMPLE_RATE)
        else:
            audio_16k = audio
        
        # Prepare message
        msg = message if message is not None else self.message
        msg = msg.to(self.device)
        if msg.shape[0] != audio_16k.shape[0]:
            msg = msg.expand(audio_16k.shape[0], -1)
        
        # Generate watermark at 16kHz
        with torch.no_grad():
            watermark_16k = self.generator.get_watermark(audio_16k, self.AUDIOSEAL_SAMPLE_RATE, message=msg)
        
        # Resample watermark back to original sample rate
        if sample_rate != self.AUDIOSEAL_SAMPLE_RATE:
            watermark = self._resample(watermark_16k, self.AUDIOSEAL_SAMPLE_RATE, sample_rate)
            # Ensure same length as original
            if watermark.shape[-1] != audio.shape[-1]:
                if watermark.shape[-1] > audio.shape[-1]:
                    watermark = watermark[..., :audio.shape[-1]]
                else:
                    watermark = torch.nn.functional.pad(
                        watermark, (0, audio.shape[-1] - watermark.shape[-1])
                    )
            # Re-fetch original audio at original sample rate
            audio = self._resample(audio_16k, self.AUDIOSEAL_SAMPLE_RATE, sample_rate)
            if audio.shape[-1] != original_shape[-1] if len(original_shape) > 0 else True:
                # Adjust to match original length
                target_len = original_shape[-1] if original_shape else watermark.shape[-1]
                if audio.shape[-1] > target_len:
                    audio = audio[..., :target_len]
                    watermark = watermark[..., :target_len]
        else:
            watermark = watermark_16k
        
        # Add watermark to audio
        watermarked = audio + watermark
        
        # Prevent clipping
        max_val = watermarked.abs().max()
        if max_val > 1.0:
            watermarked = watermarked / max_val
        
        # Restore original shape
        if len(original_shape) == 1:
            watermarked = watermarked.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            watermarked = watermarked.squeeze(0)
        
        # Restore device and dtype
        watermarked = watermarked.to(device=original_device, dtype=original_dtype)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            watermarked = watermarked.numpy()
        
        return watermarked
    
    def detect(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 24000,
        threshold: float = 0.5,
    ) -> WatermarkResult:
        """Detect watermark in audio.
        
        Args:
            audio: Input audio to check for watermark
            sample_rate: Sample rate of input audio
            threshold: Detection threshold (0.0-1.0)
            
        Returns:
            WatermarkResult with detection status, confidence, and decoded message
        """
        if not AUDIOSEAL_AVAILABLE:
            return WatermarkResult(
                detected=False,
                confidence=0.0,
                message=None,
                frame_probabilities=None,
            )
        
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        
        audio = audio.float()
        
        # Handle different input shapes
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.ndim == 2:
            audio = audio.unsqueeze(0)
        
        audio = audio.to(self.device)
        
        # Resample to 16kHz
        if sample_rate != self.AUDIOSEAL_SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, self.AUDIOSEAL_SAMPLE_RATE)
        
        # Detect watermark
        with torch.no_grad():
            result, message = self.detector(audio, self.AUDIOSEAL_SAMPLE_RATE)
        
        # result shape: (batch, 2, frames) - probabilities for [no_watermark, watermark]
        # Get positive (watermark present) probabilities
        watermark_probs = result[:, 1, :]  # (batch, frames)
        
        # Calculate overall confidence as mean of frame probabilities
        confidence = watermark_probs.mean().item()
        
        # Detection based on threshold
        detected = confidence > threshold
        
        return WatermarkResult(
            detected=detected,
            confidence=confidence,
            message=message.cpu() if message is not None else None,
            frame_probabilities=watermark_probs.cpu(),
        )
    
    def verify(self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int = 24000) -> bool:
        """Quick verification that audio contains KugelAudio watermark.
        
        Args:
            audio: Audio to verify
            sample_rate: Sample rate of audio
            
        Returns:
            True if watermark detected with high confidence
        """
        result = self.detect(audio, sample_rate)
        return result.detected and result.confidence > 0.6


class WatermarkPostProcessor:
    """Post-processor that automatically adds watermarks to generated audio.
    
    Designed to be integrated into the generation pipeline to ensure
    all generated audio is watermarked transparently.
    
    Example:
        >>> post_processor = WatermarkPostProcessor()
        >>> # In generation pipeline:
        >>> audio = model.generate(...)
        >>> audio = post_processor(audio)  # Watermark added automatically
    """
    
    def __init__(
        self,
        enabled: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        sample_rate: int = 24000,
    ):
        self.enabled = enabled
        self.sample_rate = sample_rate
        self._watermark = None
        self._device = device
    
    @property
    def watermark(self) -> AudioWatermark:
        """Lazy initialization of watermark model."""
        if self._watermark is None:
            self._watermark = AudioWatermark(device=self._device)
        return self._watermark
    
    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Add watermark to audio if enabled."""
        if not self.enabled:
            return audio
        
        sr = sample_rate or self.sample_rate
        return self.watermark.embed(audio, sample_rate=sr)
    
    def disable(self):
        """Disable watermarking."""
        self.enabled = False
    
    def enable(self):
        """Enable watermarking."""
        self.enabled = True


def is_watermarked(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 24000,
    threshold: float = 0.5,
) -> bool:
    """Convenience function to check if audio is watermarked.
    
    Args:
        audio: Audio to check
        sample_rate: Sample rate of audio
        threshold: Detection threshold
        
    Returns:
        True if watermark detected
    """
    watermark = AudioWatermark()
    result = watermark.detect(audio, sample_rate, threshold)
    return result.detected
