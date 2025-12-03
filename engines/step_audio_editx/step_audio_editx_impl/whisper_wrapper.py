import logging
import torch
import torchaudio
from transformers import pipeline


class WhisperWrapper:
    """Simplified Whisper ASR wrapper"""

    def __init__(self, model_id="openai/whisper-large-v3"):
        """
        Initialize WhisperWrapper

        Args:
            model_id: Whisper model ID, default uses openai/whisper-large-v3
        """
        self.logger = logging.getLogger(__name__)
        self.model = None

        try:
            self.model = pipeline("automatic-speech-recognition", model=model_id)
            self.logger.info(f"✓ Whisper model loaded successfully: {model_id}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load Whisper model: {e}")
            raise

    def __call__(self, audio_input):
        """
        Audio to text transcription

        Args:
            audio_input: Audio file path or audio tensor

        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")

        try:
            # Load audio
            if isinstance(audio_input, str):
                # Audio file path
                audio, audio_sr = torchaudio.load(audio_input)
                audio = torchaudio.functional.resample(audio, audio_sr, 16000)
                # Handle stereo to mono conversion (pipeline may not handle this)
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)  # Convert stereo to mono by averaging
                # Convert to numpy and squeeze
                audio = audio.squeeze(0).numpy()
            elif isinstance(audio_input, torch.Tensor):
                # Tensor input
                audio = audio_input.cpu()
                audio = torchaudio.functional.resample(audio, audio_sr, 16000)
                # Handle stereo to mono conversion
                if audio.ndim > 1 and audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                audio = audio.squeeze().numpy()
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

            # Transcribe
            result = self.model(audio)
            text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()

            self.logger.debug(f"Transcription result: {text}")
            return text

        except Exception as e:
            self.logger.error(f"Audio transcription failed: {e}")
            return ""

    def is_available(self):
        """Check if whisper model is available"""
        return self.model is not None