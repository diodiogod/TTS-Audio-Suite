import torch
import torchaudio
import numpy as np
import tempfile
import importlib.util
import threading
import time
import queue

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

class ChatterBoxVoiceCapture:
    @classmethod
    def NAME(cls):
        if not SOUNDDEVICE_MODULE_AVAILABLE:
            return "🎙️ ChatterBox Voice Capture (diogod) - PortAudio Required"
        return "🎙️ ChatterBox Voice Capture (diogod)"
    
    @classmethod
    def INPUT_TYPES(cls):
        if not SOUNDDEVICE_MODULE_AVAILABLE:
            return {
                "required": {
                    "error_message": (["PortAudio library not found. Install with: sudo apt-get install portaudio19-dev (Linux) or brew install portaudio (macOS)"], {"default": "PortAudio library not found. Install with: sudo apt-get install portaudio19-dev (Linux) or brew install portaudio (macOS)"}),
                }
            }

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
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("voice_audio",)
    FUNCTION = "capture_voice_audio"
    CATEGORY = "TTS Audio Suite/🎵 Audio Processing"

    def capture_voice_audio(self, **kwargs):
        sd, sounddevice_error = _load_sounddevice()
        if sd is None:
            print(f"❌ ChatterBox Voice Capture error: {sounddevice_error}")
            print("📋 Install PortAudio to enable voice recording:")
            print("   Linux: sudo apt-get install portaudio19-dev")
            print("   macOS: brew install portaudio")
            print("   Windows: Usually bundled with sounddevice")
            # Return empty audio tensor
            return (torch.zeros(1, 1, 24000),)
        
        # Extract parameters with defaults for graceful fallback
        voice_device = kwargs.get('voice_device', '')
        voice_sample_rate = kwargs.get('voice_sample_rate', 44100)
        voice_max_recording_time = kwargs.get('voice_max_recording_time', 10.0)
        voice_volume_gain = kwargs.get('voice_volume_gain', 1.0)
        voice_silence_threshold = kwargs.get('voice_silence_threshold', 0.02)
        voice_silence_duration = kwargs.get('voice_silence_duration', 2.0)
        voice_auto_normalize = kwargs.get('voice_auto_normalize', True)
        voice_trigger = kwargs.get('voice_trigger', 0)
        
        print(f"🎤 Starting ChatterBox Voice Capture...")
        print(f"Settings: max_time={voice_max_recording_time}s, volume_gain={voice_volume_gain}x, silence_threshold={voice_silence_threshold}, silence_duration={voice_silence_duration}s, rate={voice_sample_rate}")
        print(f"Auto-normalize: {'ON' if voice_auto_normalize else 'OFF'}")
        
        # Parse device
        try:
            device_index, resolved_device_info, used_default_device = _resolve_input_device(sd, voice_device)
        except Exception as e:
            print(f"⚠️  Device selection error: {e}")
            device_index = None
            resolved_device_info = None
            used_default_device = True

        resolved_device_name = ""
        resolved_default_sample_rate = None
        if resolved_device_info:
            resolved_device_name = str(resolved_device_info.get("name", "")).strip()
            resolved_default_sample_rate = resolved_device_info.get("default_samplerate")

        if used_default_device:
            if resolved_device_name:
                print(f"🎙️ Using resolved system default input device: {resolved_device_name}")
            else:
                print("🎙️ Using system default input device")
        elif resolved_device_name:
            print(f"🎙️ Using selected input device: {resolved_device_name}")

        if used_default_device and resolved_default_sample_rate:
            try:
                native_sample_rate = int(round(float(resolved_default_sample_rate)))
            except Exception:
                native_sample_rate = None

            if native_sample_rate and native_sample_rate > 0 and native_sample_rate != voice_sample_rate:
                print(
                    f"ℹ️ System default input device native sample rate is {native_sample_rate} Hz; "
                    f"using that instead of requested {voice_sample_rate} Hz"
                )
                voice_sample_rate = native_sample_rate

        print(f"🔊 Opening voice stream...")

        # Recording parameters
        chunk_size = int(voice_sample_rate * 0.1)  # 100ms chunks
        max_chunks = int(voice_max_recording_time * 10)  # 10 chunks per second
        
        voice_data = queue.Queue()
        recording_active = threading.Event()
        recording_active.set()

        def voice_callback(indata, frames, time, status):
            if status:
                print(f"⚠️  Voice stream status: {status}")
            if recording_active.is_set():
                voice_data.put(indata.copy())

        try:
            # Start recording stream
            with sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=voice_sample_rate,
                blocksize=chunk_size,
                callback=voice_callback,
                dtype=np.float32
            ):
                print(f"🔴 Voice recording in progress...")
                
                voice_chunks = []
                chunk_count = 0
                silence_start = None
                max_level_seen = 0.0
                exit_reason = "max_time_reached"
                
                start_time = time.time()
                
                while chunk_count < max_chunks and recording_active.is_set():
                    try:
                        # Get chunk with timeout
                        chunk = voice_data.get(timeout=0.2)
                        voice_chunks.append(chunk)
                        chunk_count += 1
                        
                        # Apply volume gain
                        gained_chunk = chunk * voice_volume_gain
                        current_level = np.max(np.abs(gained_chunk))
                        max_level_seen = max(max_level_seen, current_level)
                        
                        elapsed_time = time.time() - start_time
                        
                        # Progress logging every 2 seconds
                        if chunk_count % 20 == 0:  # Every 2 seconds instead of every second
                            avg_level = np.sqrt(np.mean(gained_chunk**2))
                            silence_status = "🔇 QUIET" if current_level < voice_silence_threshold else "🔊 SOUND"
                            print(f"📊 Voice Level: peak={current_level:.3f}, avg={avg_level:.3f}, max_seen={max_level_seen:.3f}, time={elapsed_time:.1f}s, chunks={chunk_count} | {silence_status} (threshold={voice_silence_threshold})")
                            
                            # Warn if levels are problematic
                            if max_level_seen > 0.95:
                                print("⚠️  Voice audio is clipping! Consider reducing voice_volume_gain.")
                            elif max_level_seen < 0.01:
                                print("⚠️  Voice audio is very quiet. Consider increasing voice_volume_gain.")
                            elif current_level < voice_silence_threshold:
                                print(f"💡 TIP: Currently below silence threshold. Voice silence detection active.")
                            elif current_level > voice_silence_threshold and current_level < voice_silence_threshold * 2:
                                print(f"💡 TIP: Close to silence threshold. Consider adjusting to {current_level + 0.005:.3f}")
                        
                        # Check for silence (using gained audio for accurate detection)
                        silence_level = np.max(np.abs(gained_chunk))
                        if silence_level < voice_silence_threshold:
                            if silence_start is None:
                                silence_start = time.time()
                                print(f"🔇 Voice silence started (level={silence_level:.4f} < {voice_silence_threshold})")
                            else:
                                silence_elapsed = time.time() - silence_start
                                if silence_elapsed >= voice_silence_duration:
                                    exit_reason = "voice_silence_detected"
                                    print(f"🔇 Detected {voice_silence_duration} seconds of voice silence, stopping...")
                                    print(f"🛑 VOICE SILENCE BREAK: Exiting recording loop now!")
                                    break
                                elif chunk_count % 5 == 0:  # Show progress every 500ms during silence
                                    print(f"🔇 Voice Silence: {silence_elapsed:.1f}s / {voice_silence_duration}s (level={silence_level:.4f})")
                        else:
                            if silence_start is not None:
                                print(f"🔊 Voice sound detected, resetting silence timer (level={silence_level:.4f} > {voice_silence_threshold})")
                            silence_start = None
                            
                    except queue.Empty:
                        continue
                    except KeyboardInterrupt:
                        exit_reason = "user_interrupted"
                        break

            recording_active.clear()
            
        except Exception as e:
            print(f"❌ Voice recording error: {e}")
            return (torch.zeros(1, 1, voice_sample_rate),)

        print(f"⏰ Voice recording stopped: {exit_reason}")
        print(f"🛑 Voice recording loop completed!")
        
        if not voice_chunks:
            print("⚠️  No voice audio captured!")
            return (torch.zeros(1, 1, voice_sample_rate),)
        
        # Process recorded audio
        print(f"📊 Voice recording duration: {len(voice_chunks) * 0.1:.1f}s, chunks collected: {len(voice_chunks)}")
        
        # Combine chunks
        voice_recording = np.concatenate(voice_chunks, axis=0).flatten()
        
        # Apply gain
        voice_recording = voice_recording * voice_volume_gain
        
        # Calculate final levels
        final_peak = np.max(np.abs(voice_recording))
        final_avg = np.sqrt(np.mean(voice_recording**2))
        print(f"⚙️  Processing voice recording...")
        print(f"📊 Final voice levels: peak={final_peak:.3f}, avg={final_avg:.3f}")
        
        # Auto-normalize if enabled
        if voice_auto_normalize and final_peak > 0:
            # Target peak at 0.8 to leave some headroom
            normalize_factor = 0.8 / final_peak
            voice_recording = voice_recording * normalize_factor
            final_peak_after = np.max(np.abs(voice_recording))
            print(f"🔧 Voice auto-normalized: {normalize_factor:.3f}x (peak: {final_peak:.3f} → {final_peak_after:.3f})")
        
        # Convert to tensor format expected by ComfyUI
        voice_tensor = torch.from_numpy(voice_recording).float().unsqueeze(0).unsqueeze(0)
        
        print(f"✅ Voice capture complete: {voice_tensor.shape[1] / voice_sample_rate:.1f}s, peak={final_peak:.3f}, avg={final_avg:.3f}")
        
        # Save to temp file for debugging
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            torchaudio.save(temp_path, voice_tensor.squeeze(0), voice_sample_rate)
            print(f"💾 Voice recording saved to: {temp_path}")
            
        except Exception as e:
            print(f"⚠️  Could not save voice recording: {e}")
        
        return ({
            "waveform": voice_tensor,
            "sample_rate": voice_sample_rate
        },)
