"""Microphone capture outside workflow execution; completed clips persist in input/."""

import importlib.util
import json
import math
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from utils.security.path_access import child_path

# Do not import sounddevice or enumerate devices at startup.
# On some Windows systems PortAudio can hang indefinitely during import/device probing,
# which blocks ComfyUI before the server starts.
SOUNDDEVICE_MODULE_AVAILABLE = importlib.util.find_spec("sounddevice") is not None


def _load_sounddevice():
    """Import sounddevice only when the node is actually used."""
    if not SOUNDDEVICE_MODULE_AVAILABLE:
        return None, "sounddevice package not installed"

    try:
        import sounddevice as sd
        return sd, None
    except Exception as e:
        return None, str(e)


def _get_first_input_device(sd):
    """Return the first available input device as a safe fallback."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            return i, device
    return None, None


def _get_default_input_device(sd):
    """Resolve PortAudio's default input device explicitly."""
    try:
        default_device = getattr(sd.default, "device", None)
        if isinstance(default_device, (list, tuple)) and len(default_device) > 0:
            input_index = default_device[0]
        else:
            input_index = default_device

        if input_index is not None:
            input_index = int(input_index)
            if input_index >= 0:
                return input_index, sd.query_devices(input_index, "input")
    except Exception:
        pass

    try:
        default_info = sd.query_devices(kind="input")
        default_name = str(default_info.get("name", "")).strip()
        if default_name:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device.get("max_input_channels", 0) <= 0:
                    continue
                if str(device.get("name", "")).strip() == default_name:
                    return i, device
    except Exception:
        pass

    return _get_first_input_device(sd)


def _resolve_input_device(sd, requested_device_name):
    """Resolve an input device selection at runtime."""
    requested = (requested_device_name or "").strip()
    if not requested:
        device_index, device_info = _get_default_input_device(sd)
        return device_index, device_info, True

    normalized = requested.lower()
    if normalized in {"default", "system default", "system default input device", "auto"}:
        device_index, device_info = _get_default_input_device(sd)
        return device_index, device_info, True

    legacy_suffix = " - input"
    if normalized.endswith(legacy_suffix):
        normalized = normalized[:-len(legacy_suffix)].strip()

    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"⚠️  Could not enumerate input devices, using system default: {e}")
        device_index, device_info = _get_default_input_device(sd)
        return device_index, device_info, True

    for i, device in enumerate(devices):
        device_name = str(device.get("name", ""))
        if device.get("max_input_channels", 0) > 0 and normalized in device_name.lower():
            return i, device, False

    print(f"⚠️  Requested input device '{requested}' not found, using system default.")
    device_index, device_info = _get_default_input_device(sd)
    return device_index, device_info, True


def recording_path(recording):
    import folder_paths

    if not re.fullmatch(r"[a-f0-9]{32}", str(recording or "")):
        raise ValueError("Record a clip with Start Recording before running the workflow.")
    return Path(child_path(folder_paths.get_input_directory(), "voice_capture", f"{recording}.wav"))


def input_devices():
    # Isolate PortAudio probing so a driver hang cannot block the server.
    script = """
import json
import sounddevice as sd
print(json.dumps(list(dict.fromkeys(
    str(device['name']).strip() for device in sd.query_devices()
    if device.get('max_input_channels', 0) > 0
))))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=8, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Could not enumerate microphone devices.")
    return json.loads(result.stdout)


def capture_settings(data):
    settings = {"voice_device": str(data.get("voice_device", "") or ""),
                "voice_auto_normalize": bool(data.get("voice_auto_normalize", True))}
    for name, default, lower, upper in (
        ("voice_sample_rate", 44100, 8000, 96000),
        ("voice_max_recording_time", 10.0, 1.0, 300.0),
        ("voice_volume_gain", 1.0, 0.1, 10.0),
        ("voice_silence_threshold", 0.02, 0.001, 0.1),
        ("voice_silence_duration", 2.0, 0.5, 10.0),
    ):
        value = float(data.get(name, default))
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}.")
        settings[name] = int(value) if name == "voice_sample_rate" else value
    return settings


class CaptureSession:
    def __init__(self, settings):
        self.recording = uuid.uuid4().hex
        self.settings = settings
        self.stop_event = threading.Event()
        self.state = "starting"
        self.error = ""
        self.duration = 0.0
        self.sample_rate = settings["voice_sample_rate"]
        self.thread = threading.Thread(target=self._record, daemon=True,
                                       name="tts-voice-capture")

    def snapshot(self):
        return {"recording": self.recording, "state": self.state,
                "duration": self.duration, "sample_rate": self.sample_rate,
                "error": self.error}

    def _record(self):
        try:
            import soundfile as sf

            sd, error = _load_sounddevice()
            if sd is None:
                raise RuntimeError(error)
            device, info, default = _resolve_input_device(sd, self.settings["voice_device"])
            if default and info and info.get("default_samplerate"):
                self.sample_rate = int(round(info["default_samplerate"]))
            if self.stop_event.is_set():
                raise RuntimeError("Recording stopped before the microphone opened.")

            chunks = []
            incoming = queue.Queue()
            blocksize = max(1, int(self.sample_rate * 0.1))
            max_time = self.settings["voice_max_recording_time"]

            def callback(indata, frames, callback_time, status):
                if not self.stop_event.is_set():
                    incoming.put(indata.copy())

            with sd.InputStream(device=device, channels=1, samplerate=self.sample_rate,
                                blocksize=blocksize, callback=callback, dtype="float32"):
                self.state = "recording"
                started = time.monotonic()
                silence_started = None
                frames = 0
                while not self.stop_event.is_set() and time.monotonic() - started < max_time:
                    try:
                        chunk = incoming.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    chunks.append(chunk)
                    frames += len(chunk)
                    self.duration = frames / self.sample_rate
                    level = np.max(np.abs(chunk)) * self.settings["voice_volume_gain"]
                    if level < self.settings["voice_silence_threshold"]:
                        if silence_started is None:
                            silence_started = time.monotonic()
                        elif time.monotonic() - silence_started >= self.settings["voice_silence_duration"]:
                            break
                    else:
                        silence_started = None

            if not chunks:
                raise RuntimeError("No microphone audio was captured. Check the input device and try again.")
            self.state = "saving"
            audio = np.concatenate(chunks, axis=0) * self.settings["voice_volume_gain"]
            peak = float(np.max(np.abs(audio)))
            if self.settings["voice_auto_normalize"] and peak > 0:
                audio *= 0.8 / peak
            path = recording_path(self.recording)
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), audio, self.sample_rate, subtype="FLOAT")
            # Publish ready only after closing the complete file.
            self.state = "ready"
        except Exception as error:
            self.error = str(error)
            self.state = "error"


_capture_lock = threading.Lock()
_current_capture = None


def start_capture(data):
    global _current_capture
    settings = capture_settings(data)
    with _capture_lock:
        if _current_capture is not None and _current_capture.thread.is_alive():
            raise RuntimeError("Another microphone recording is already in progress.")
        _current_capture = CaptureSession(settings)
        _current_capture.thread.start()
        return _current_capture.snapshot()


def capture_status(recording, *, stop=False):
    with _capture_lock:
        if _current_capture is None or _current_capture.recording != recording:
            raise ValueError("Recording session no longer exists. Start a new recording.")
        if stop:
            _current_capture.stop_event.set()
        return _current_capture.snapshot()
