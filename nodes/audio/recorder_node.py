"""Voice Capture outputs a saved recording; microphone controls run independently."""

from utils.audio.processing import AudioProcessingUtils
from utils.audio.trim import trim_audio
from utils.audio.voice_capture import SOUNDDEVICE_MODULE_AVAILABLE, recording_path

class ChatterBoxVoiceCapture:
    @classmethod
    def NAME(cls):
        if not SOUNDDEVICE_MODULE_AVAILABLE:
            return "🎙️ ChatterBox Voice Capture (diogod) - PortAudio Required"
        return "🎙️ ChatterBox Voice Capture (diogod)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_device": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional input device name match. Leave empty to use the system default device. Device lookup is deferred until recording so ComfyUI startup does not block on PortAudio."
                }),
                "voice_sample_rate": ("INT", {
                    "default": 44100,
                    "min": 8000,
                    "max": 96000,
                    "step": 1
                }),
                "voice_max_recording_time": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 300.0,
                    "step": 0.1
                }),
                "voice_volume_gain": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "voice_silence_threshold": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001
                }),
                "voice_silence_duration": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1
                }),
                "voice_auto_normalize": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "voice_trigger": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999
                }),
                "recording": ("STRING", {"default": "", "tooltip": "Saved microphone recording selected by the preview controls."}),
                "trim_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
                "trim_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("voice_audio",)
    FUNCTION = "capture_voice_audio"
    CATEGORY = "TTS Audio Suite/🎵 Audio Processing"

    @classmethod
    def IS_CHANGED(cls, recording="", **kwargs):
        # Saved clips are reusable. File removal/replacement must invalidate cached audio.
        try:
            stat = recording_path(recording).stat()
            return stat.st_mtime_ns, stat.st_size
        except (ValueError, OSError):
            return "missing recording"

    def capture_voice_audio(self, recording="", trim_start=0.0, trim_end=0.0, **kwargs):
        path = recording_path(recording)
        if not path.is_file():
            raise FileNotFoundError("Voice Capture recording is missing. Record a new clip before running the workflow.")
        waveform, sample_rate = AudioProcessingUtils.safe_load_audio(str(path))
        audio, _ = trim_audio({"waveform": waveform, "sample_rate": sample_rate},
                              trim_start, trim_end, component="Voice Capture")
        return ({"waveform": audio["waveform"].unsqueeze(0), "sample_rate": sample_rate},)
