import os
import sys

# Smart numba compatibility for vocal separation
from utils.compatibility import setup_numba_compatibility
setup_numba_compatibility(quick_startup=True, verbose=False)

# NumPy 2.x compatibility fix
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.complex = complex
    np.bool = bool
    
import audio_separator.separator as uvr

# Add engine path for imports
import sys
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(os.path.dirname(current_dir))
rvc_impl_path = os.path.join(engines_dir, "engines", "rvc", "impl")
if rvc_impl_path not in sys.path:
    sys.path.insert(0, rvc_impl_path)

# AnyType for flexible input types
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

from rvc_audio import audio_to_bytes, save_input_audio, load_input_audio, get_audio
import folder_paths
from rvc_utils import get_filenames, get_hash, get_optimal_torch_device
from lib import karafan
from rvc_downloader import KARAFAN_MODELS, MDX_MODELS, RVC_DOWNLOAD_LINK, VR_MODELS, ZFTURBO_MODELS, ZFTURBO_DOWNLOAD_LINK, MELBAND_MODELS, MELBAND_DOWNLOAD_LINK, download_file

# Define paths
BASE_CACHE_DIR = folder_paths.get_temp_directory()
BASE_MODELS_DIR = folder_paths.models_dir

temp_path = folder_paths.get_temp_directory()
cache_dir = os.path.join(BASE_CACHE_DIR,"uvr")
device = get_optimal_torch_device()
is_half = True

def _find_uvr_model_in_cache(model_filename: str) -> str:
    """
    Find the specific UVR/audio separation model in HuggingFace cache
    Only checks for models that actually come from HuggingFace (RVC-Studio repo)
    
    Args:
        model_filename: Name of the model file (e.g., "UVR-MDX-NET-vocal_FT.onnx")
        
    Returns:
        Path to the exact cached model if found, None otherwise
    """
    try:
        # Get HuggingFace cache directory
        cache_home = os.environ.get('HUGGINGFACE_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
        
        # Only check the actual HuggingFace repo we download from
        # RVC_DOWNLOAD_LINK = 'https://huggingface.co/datasets/SayanoAI/RVC-Studio/resolve/main/'
        # Note: ZFTurbo models come from GitHub releases, so no HF cache for those
        repo = "SayanoAI/RVC-Studio"
        cache_folder_name = f"models--{repo.replace('/', '--')}"
        cache_path = os.path.join(cache_home, cache_folder_name)
        
        if os.path.exists(cache_path):
            # Look for the snapshots directory
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                try:
                    snapshots = os.listdir(snapshots_dir)
                    if snapshots:
                        # Check each snapshot for the specific model file
                        for snapshot in snapshots:
                            snapshot_path = os.path.join(snapshots_dir, snapshot)
                            # Model could be in a subfolder (like MDXNET/UVR-MDX-NET-vocal_FT.onnx)
                            # or directly in the snapshot
                            model_path = os.path.join(snapshot_path, model_filename)
                            if os.path.exists(model_path):
                                return model_path
                except OSError:
                    pass
        
        return None
    except Exception as e:
        print(f"⚠️ Error checking HuggingFace cache for UVR model '{model_filename}': {e}")
        return None

class VocalRemovalNode:
    
    @classmethod
    def NAME(cls):
        return "🤐 Noise or Vocal Removal"
 
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):

        # Add ZFTurbo SOTA models to the list
        zfturbo_model_names = [model_path for _, model_path in ZFTURBO_MODELS]

        # Search both TTS and legacy paths for models
        tts_models = get_filenames(root=os.path.join(BASE_MODELS_DIR, "TTS"),format_func=lambda x: f"{os.path.basename(os.path.dirname(x))}/{os.path.basename(x)}",name_filters=["UVR","MDX","karafan","SCNET","MDX23C","MELBAND"])
        legacy_models = get_filenames(root=BASE_MODELS_DIR,format_func=lambda x: f"{os.path.basename(os.path.dirname(x))}/{os.path.basename(x)}",name_filters=["UVR","MDX","karafan","SCNET","MDX23C","MELBAND"])

        model_list = (MDX_MODELS + VR_MODELS + KARAFAN_MODELS + zfturbo_model_names + MELBAND_MODELS + tts_models + legacy_models)
        model_list = list(set(model_list)) # dedupe

        # Filter out non-model files (JSON configs, etc.)
        model_extensions = ['.pth', '.ckpt', '.onnx', '.pt', '.safetensors']
        model_list = [model for model in model_list if any(model.lower().endswith(ext) for ext in model_extensions)]

        # Sort by category (folder name) then alphabetically within each category
        def sort_key(model_path):
            parts = model_path.split('/')
            category = parts[0].upper() if len(parts) > 1 else "OTHER"
            filename = parts[-1]
            return (category, filename)

        model_list = sorted(model_list, key=sort_key)

        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio for vocal/instrumental separation. Standard ComfyUI AUDIO format."
                }),
                "model": (model_list,{
                    "default": "MELBAND/MelBandRoformer_fp16.safetensors",
                    "tooltip": """🎵 AI AUDIO SEPARATION & PROCESSING

🏆 RECOMMENDED MODELS:

🎯 MELBAND ROFORMER (State-of-the-Art):
• MelBandRoformer_fp16.safetensors - ⭐ DEFAULT: Fast & High Quality (456MB)
• MelBandRoformer_fp32.safetensors - 💎 Maximum Quality (913MB)  
• denoise_mel_band_roformer_sdr_27.99.ckpt - 🥇 BEST Denoising (27.99 SDR!)
• denoise_mel_band_roformer_aggressive_sdr_27.97.ckpt - 💪 Aggressive Denoising

📀 STANDARD VOCAL SEPARATION:
• UVR-MDX-NET-vocal_FT.onnx - 🔧 Reliable (needs GPU for speed)
• model_bs_roformer_ep_317_sdr_12.9755.ckpt - 🎵 High Quality (12.98 SDR)
• HP5-vocals+instrumentals.pth - 🏠 Beginner Friendly

🔧 SPECIAL PURPOSE:
• UVR-DeNoise - NOISE REMOVAL: "remaining" = clean audio ✅
• UVR-DeEcho-DeReverb - ECHO/REVERB REMOVAL: "remaining" = dry audio ✅

⚠️ EXPERIMENTAL (Issues):
• model_vocals_mdx23c_sdr_10.17.ckpt - MDX23C architecture (tensor errors)
• model_scnet_xl_ihf_sdr_10.08.ckpt - SCNet architecture (audio artifacts)

💡 QUICK RECOMMENDATIONS:
• General Use: MelBandRoformer_fp16
• Best Denoising: MELBAND models (27+ SDR!)
• Fast Processing: MelBandRoformer_fp16
• Maximum Quality: MelBandRoformer_fp32

📖 Complete guide: docs/VOCAL_REMOVAL_GUIDE.md"""
                }),
            },
            "optional": {
                "use_cache": ("BOOLEAN",{
                    "default": True,
                    "tooltip": """🚀 CACHING SYSTEM

Enables intelligent caching of separation results for faster processing:
• ✅ ON (Recommended): Saves results to disk, dramatically speeds up repeated processing of same audio/model combinations
• ❌ OFF: Always processes from scratch, uses more time but ensures fresh results

💡 Cache includes model, aggressiveness, format, and audio content in hash
🔄 Automatically invalidates when any parameter changes
💾 Cached files stored in organized folder structure for easy management"""
                }),
                "aggressiveness":("INT",{
                    "default": 10, 
                    "min": 0, #Minimum value
                    "max": 20, #Maximum value
                    "step": 1, #Slider's step
                    "display": "slider",
                    "tooltip": """🎚️ SEPARATION AGGRESSIVENESS (0-20)

Controls separation strength for VR architecture models (HP5, DeNoise, DeEcho, etc.):

📊 RECOMMENDED VALUES:
• 0-5: Gentle separation, preserves more original audio quality
• 6-10: ⭐ BALANCED (Default: 10) - Good separation with minimal artifacts
• 11-15: Aggressive separation, may introduce artifacts but better isolation
• 16-20: Maximum aggression, highest separation but potential quality loss

🎯 USE CASES:
• 🎤 Karaoke Creation: 12-15 (more aggressive)
• 🎵 Vocal Extraction: 8-12 (balanced)
• 🎼 Preserve Music Quality: 5-8 (gentle)
• 🔧 Problem Audio: 15-20 (maximum effort)

⚠️ NOTE: Only affects VR Architecture models (HP5, DeNoise, DeEcho). Advanced models (UVR-MDX-NET, bs_roformer, MDX23C) ignore this setting.
💡 Higher values = stronger vocal/instrumental separation but may affect audio quality"""
                }),
                "format":(["wav", "flac", "mp3"],{
                    "default": "flac",
                    "tooltip": """🎵 OUTPUT AUDIO FORMAT

Selects the audio format for separated stems:

🏆 QUALITY RANKING:
• 📀 FLAC: ⭐ BEST - Lossless compression, perfect quality, larger files
• 🎵 WAV: Uncompressed, perfect quality, largest files  
• 🎧 MP3: Lossy compression, smaller files, slight quality loss

💼 PROFESSIONAL USE: FLAC (default)
🚀 FAST WORKFLOW: MP3 (smaller files, faster I/O)
🎯 MAXIMUM QUALITY: WAV (no compression)

📊 FILE SIZE COMPARISON (typical 4-minute song):
• WAV: ~40MB per stem
• FLAC: ~20MB per stem  
• MP3: ~4MB per stem

💡 All formats support the full separation quality - format only affects storage and compatibility"""
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("extracted voice/noise/echo", "remaining")

    FUNCTION = "split"

    CATEGORY = "TTS Audio Suite/🎵 Audio Processing"

    def split(self, audio, model, use_cache=True, aggressiveness=10, format='flac'):
        
        # Python 3.13 compatibility: Smart model substitution for known incompatible models
        if sys.version_info >= (3, 13):
            incompatible_models = {
                "UVR-DeNoise.pth": "UVR-DeEcho-DeReverb.pth",
                "UVR/UVR-DeNoise.pth": "UVR/UVR-DeEcho-DeReverb.pth",
                # Add more substitutions as we discover them
            }
            
            original_model = model
            for incompatible, compatible in incompatible_models.items():
                if incompatible.lower() in model.lower():
                    model = compatible
                    print(f"🔄 Python 3.13 compatibility: Substituting '{os.path.basename(original_model)}' → '{os.path.basename(model)}'")
                    print(f"💡 Reason: '{os.path.basename(original_model)}' requires Audio-Separator features not available in Python 3.13")
                    break
        
        filename = os.path.basename(model)
        subfolder = os.path.dirname(model)
        
        # Try TTS organization first, then legacy
        tts_model_path = os.path.join(BASE_MODELS_DIR, "TTS", subfolder, filename)
        legacy_model_path = os.path.join(BASE_MODELS_DIR, subfolder, filename)
        
        if os.path.isfile(tts_model_path):
            model_path = tts_model_path
        elif os.path.isfile(legacy_model_path):
            model_path = legacy_model_path
        else:
            # Model not found, will download to TTS path
            model_path = tts_model_path
        
        if not os.path.isfile(model_path):
            # Check if it's a ZFTurbo model (GitHub releases - no HF cache)
            zfturbo_model = next((download_path for download_path, model_path_check in ZFTURBO_MODELS if model_path_check == model), None)
            
            # Check if it's a MelBand model from Kijai's repo
            melband_model = next((m for m in MELBAND_MODELS if m == model), None)
            
            if zfturbo_model:
                # ZFTurbo models come from GitHub releases, no cache to check
                download_link = f"{ZFTURBO_DOWNLOAD_LINK}{zfturbo_model}"
                print(f"📥 Downloading SOTA model from ZFTurbo repository: {filename}")
                params = model_path, download_link
                if download_file(params): print(f"✅ Successfully downloaded: {model_path}")
            elif melband_model:
                # MelBand models from Kijai's HuggingFace repo
                download_link = f"{MELBAND_DOWNLOAD_LINK}{filename}"
                print(f"📥 Downloading MelBandRoFormer model from Kijai's repository: {filename}")
                params = model_path, download_link
                if download_file(params): print(f"✅ Successfully downloaded: {model_path}")
            else:
                # RVC Studio models - check HuggingFace cache first
                cache_path = _find_uvr_model_in_cache(model)
                if cache_path:
                    print(f"💾 Using HuggingFace cache for UVR model '{model}': {cache_path}")
                    model_path = cache_path  # Use cached model instead of downloading
                else:
                    # Download from RVC Studio
                    download_link = f"{RVC_DOWNLOAD_LINK}{model}"
                    print(f"📥 Downloading model from RVC Studio: {filename}")
                    params = model_path, download_link
                    if download_file(params): print(f"✅ Successfully downloaded: {model_path}")
        
        input_audio = get_audio(audio)
        hash_name = get_hash(model, aggressiveness, format, audio_to_bytes(*input_audio))
        audio_path = os.path.join(temp_path,"uvr",f"{hash_name}.wav")
        primary_path = os.path.join(cache_dir,hash_name,f"primary.{format}")
        secondary_path = os.path.join(cache_dir,hash_name,f"secondary.{format}")
        primary=secondary=None

        if os.path.isfile(primary_path) and os.path.isfile(secondary_path) and use_cache:
            print(f"🚀 Using cached separation results for faster processing")
            primary = load_input_audio(primary_path)
            secondary = load_input_audio(secondary_path)
        else:
            if not os.path.isfile(audio_path):
                os.makedirs(os.path.dirname(audio_path),exist_ok=True)
                print(save_input_audio(audio_path,input_audio))
            
            print(f"🎵 Starting vocal separation with {os.path.basename(model)}")
            try: 
                if "karafan" in model_path or "MDX23C" in model_path: # try karafan implementation
                    print(f"🔧 Using Karafan separation engine for {'MDX23C' if 'MDX23C' in model_path else 'Karafan'} model")
                    
                    # Check if the actual model file exists
                    if not os.path.exists(model_path):
                        # Try alternate paths for MDX23C models
                        alternate_paths = [
                            os.path.join(BASE_MODELS_DIR, "TTS", os.path.basename(model_path)),
                            os.path.join(BASE_MODELS_DIR, "karafan", os.path.basename(model_path)),
                            os.path.join(BASE_MODELS_DIR, os.path.basename(model_path))
                        ]
                        
                        found_model = None
                        for alt_path in alternate_paths:
                            if os.path.exists(alt_path):
                                found_model = alt_path
                                break
                        
                        if found_model:
                            print(f"📍 Found MDX23C model at: {found_model}")
                            # Update the model_path to the correct location
                            model_path = found_model
                        else:
                            raise FileNotFoundError(f"MDX23C model not found: {os.path.basename(model_path)}. Please ensure the model is placed in one of: {alternate_paths}")
                    
                    primary, secondary, _ = karafan.inference.Process(audio_path,cache_dir=temp_path,format=format)
                elif "mel_band_roformer" in model_path.lower() or "melband" in model_path.lower():
                    # MelBandRoFormer implementation
                    print(f"🔧 Using MelBandRoFormer separation engine")
                    from lib.melband.mel_band_roformer import MelBandRoformer
                    import torch
                    import torch.nn.functional as F
                    from tqdm import tqdm
                    import librosa
                    from comfy.utils import load_torch_file, ProgressBar
                    from comfy import model_management as mm
                    
                    device = mm.get_torch_device()
                    offload_device = mm.unet_offload_device()
                    
                    # MelBand model configuration
                    model_config = {
                        "dim": 384,
                        "depth": 6,
                        "stereo": True,
                        "num_stems": 1,
                        "time_transformer_depth": 1,
                        "freq_transformer_depth": 1,
                        "num_bands": 60,
                        "dim_head": 64,
                        "heads": 8,
                        "attn_dropout": 0,
                        "ff_dropout": 0,
                        "flash_attn": True,
                        "dim_freqs_in": 1025,
                        "sample_rate": 44100,
                        "stft_n_fft": 2048,
                        "stft_hop_length": 441,
                        "stft_win_length": 2048,
                        "stft_normalized": False,
                        "mask_estimator_depth": 2,
                        "multi_stft_resolution_loss_weight": 1.0,
                        "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
                        "multi_stft_hop_size": 147,
                        "multi_stft_normalized": False,
                    }
                    
                    # Load MelBand model
                    melband_model = MelBandRoformer(**model_config).eval()
                    melband_model.load_state_dict(load_torch_file(model_path), strict=True)
                    
                    # Process audio with MelBand
                    audio_waveform, sample_rate = input_audio
                    
                    # Ensure stereo
                    if audio_waveform.ndim == 1:
                        audio_waveform = np.stack([audio_waveform, audio_waveform], axis=0)
                    elif audio_waveform.shape[0] == 1:
                        audio_waveform = np.repeat(audio_waveform, 2, axis=0)
                    
                    # Resample to 44100 if needed
                    if sample_rate != 44100:
                        print(f"Resampling from {sample_rate} to 44100 Hz")
                        audio_waveform = librosa.resample(audio_waveform, orig_sr=sample_rate, target_sr=44100, axis=-1)
                        sample_rate = 44100
                    
                    # Convert to torch tensor
                    audio_tensor = torch.from_numpy(audio_waveform).float()
                    
                    # Chunked processing with windowing
                    C = 352800  # Chunk size
                    N = 2
                    step = C // N
                    fade_size = C // 10
                    border = C - step
                    
                    # Pad if necessary
                    audio_length = audio_tensor.shape[1]
                    if audio_length > 2 * border and border > 0:
                        audio_tensor = F.pad(audio_tensor, (border, border), mode='reflect')
                    
                    # Create windowing array
                    def get_windowing_array(window_size, fade_size, device):
                        fadein = torch.linspace(0, 1, fade_size)
                        fadeout = torch.linspace(1, 0, fade_size)
                        window = torch.ones(window_size)
                        window[-fade_size:] *= fadeout
                        window[:fade_size] *= fadein
                        return window.to(device)
                    
                    windowing_array = get_windowing_array(C, fade_size, device)
                    
                    audio_tensor = audio_tensor.to(device)
                    vocals = torch.zeros_like(audio_tensor, dtype=torch.float32).to(device)
                    counter = torch.zeros_like(audio_tensor, dtype=torch.float32).to(device)
                    
                    total_length = audio_tensor.shape[1]
                    num_chunks = (total_length + step - 1) // step
                    
                    melband_model.to(device)
                    
                    comfy_pbar = ProgressBar(num_chunks)
                    
                    # Process chunks
                    for i in tqdm(range(0, total_length, step), desc="Processing MelBand chunks"):
                        part = audio_tensor[:, i:i + C]
                        length = part.shape[-1]
                        if length < C:
                            if length > C // 2 + 1:
                                part = F.pad(input=part, pad=(0, C - length), mode='reflect')
                            else:
                                part = F.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                        
                        x = melband_model(part.unsqueeze(0))[0]
                        
                        window = windowing_array.clone()
                        if i == 0:
                            window[:fade_size] = 1
                        elif i + C >= total_length:
                            window[-fade_size:] = 1
                        
                        vocals[..., i:i+length] += x[..., :length] * window[..., :length]
                        counter[..., i:i+length] += window[..., :length]
                        comfy_pbar.update(1)
                    
                    melband_model.to(offload_device)
                    
                    estimated_sources = vocals / counter
                    
                    # Remove padding
                    if audio_length > 2 * border and border > 0:
                        estimated_sources = estimated_sources[..., border:-border]
                    
                    # Convert back to numpy - need to preserve original audio shape
                    vocals_np = estimated_sources.cpu().numpy()
                    
                    # Use original audio tensor before padding for instrumentals calculation
                    original_audio_tensor = torch.from_numpy(audio_waveform).float()
                    if audio_length > 2 * border and border > 0:
                        # Match the cropping we did to vocals
                        instrumentals_np = (original_audio_tensor - estimated_sources.cpu()).numpy()
                    else:
                        instrumentals_np = (original_audio_tensor - estimated_sources.cpu()).numpy()
                    
                    # Save to files for caching
                    import soundfile as sf
                    os.makedirs(os.path.dirname(primary_path), exist_ok=True)
                    
                    if format == 'wav':
                        sf.write(primary_path, vocals_np.T, sample_rate, subtype='PCM_16')
                        sf.write(secondary_path, instrumentals_np.T, sample_rate, subtype='PCM_16')
                    elif format == 'flac':
                        sf.write(primary_path, vocals_np.T, sample_rate, format='FLAC', subtype='PCM_16')
                        sf.write(secondary_path, instrumentals_np.T, sample_rate, format='FLAC', subtype='PCM_16')
                    else:  # mp3
                        # For MP3, try ffmpeg conversion with fallback to WAV
                        from utils.ffmpeg_utils import convert_to_mp3_safe

                        # Save as WAV first
                        temp_wav_primary = primary_path.replace('.mp3', '_temp.wav')
                        temp_wav_secondary = secondary_path.replace('.mp3', '_temp.wav')
                        sf.write(temp_wav_primary, vocals_np.T, sample_rate, subtype='PCM_16')
                        sf.write(temp_wav_secondary, instrumentals_np.T, sample_rate, subtype='PCM_16')

                        # Convert to MP3 with fallback
                        final_primary, used_mp3_primary = convert_to_mp3_safe(temp_wav_primary, primary_path)
                        final_secondary, used_mp3_secondary = convert_to_mp3_safe(temp_wav_secondary, secondary_path)

                        # Update paths if fallback was used
                        if not used_mp3_primary:
                            primary_path = final_primary
                            print("💡 Vocals saved as WAV (ffmpeg not available for MP3 conversion)")
                        if not used_mp3_secondary:
                            secondary_path = final_secondary
                            print("💡 Instrumentals saved as WAV (ffmpeg not available for MP3 conversion)")

                        # Clean up temp files
                        try:
                            if os.path.exists(temp_wav_primary):
                                os.remove(temp_wav_primary)
                            if os.path.exists(temp_wav_secondary):
                                os.remove(temp_wav_secondary)
                        except Exception:
                            pass
                    
                    # Kijai's MelBand models output instrumentals as primary, vocals as secondary (inverted)
                    # ZFTurbo's denoise models output correctly (vocals as primary)
                    if "melbandroformer_fp" in model_path.lower():
                        # Kijai's models need swapping
                        primary = (instrumentals_np, sample_rate)  # This is actually vocals
                        secondary = (vocals_np, sample_rate)  # This is actually instrumentals
                    else:
                        # ZFTurbo's denoise models are correct
                        primary = (vocals_np, sample_rate)
                        secondary = (instrumentals_np, sample_rate)
                    
                else: # try python-audio-separator implementation
                    print(f"🔧 Using Audio-Separator engine")
                    model_dir = os.path.dirname(model_path)
                    model_name = os.path.basename(model_path)
                    vr_params={"batch_size": 4, "window_size": 512, "aggression": aggressiveness, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": "mirroring"}
                    mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 4}
                    model = uvr.Separator(model_file_dir=os.path.join(BASE_MODELS_DIR,model_dir),output_dir=temp_path,output_format=format,vr_params=vr_params,mdx_params=mdx_params)
                    model.load_model(model_name)
                    output_files = model.separate(audio_path)
                    primary = load_input_audio(os.path.join(temp_path,output_files[0]))
                    secondary = load_input_audio(os.path.join(temp_path,output_files[1]))
            except Exception as e: # try RVC implementation
                # Don't use RVC fallback for MelBand models - they're incompatible
                if "mel_band_roformer" in model_path.lower() or "melband" in model_path.lower():
                    print(f"❌ MelBandRoFormer processing failed: {e}")
                    raise RuntimeError(f"MelBandRoFormer processing failed: {e}")
                
                # Handle specific Karafan/MDX23C errors
                if ("karafan" in model_path or "MDX23C" in model_path) and ("'Separator' object has no attribute 'model'" in str(e) or "no attribute 'model'" in str(e)):
                    print(f"❌ Karafan engine failed with Audio-Separator compatibility issue")
                    print(f"🔧 Attempting fallback to RVC engine for compatible UVR models...")
                elif "MDX23C" in model_path:
                    print(f"❌ MDX23C model failed: {e}")
                    print(f"💡 MDX23C models are experimental and may have compatibility issues")
                    print(f"🔧 Attempting fallback to RVC engine...")
                
                # Skip RVC fallback for models that are architecturally incompatible
                if "MDX23C" in model_path:
                    print(f"🚫 MDX23C models are not compatible with RVC fallback engine")
                    print(f"💡 MDX23C models use specialized Karafan architecture that cannot be processed by UVR/RVC engines")
                    raise RuntimeError(f"MDX23C model '{os.path.basename(model_path)}' failed and is not compatible with fallback engines. Please use UVR-MDX-NET models instead.")
                
                print(f"⚠️ Primary engine failed, switching to RVC fallback engine...")
                print(f"💡 This is normal - downloading and using model with RVC implementation")
                
                # Get device for fallback implementation
                from comfy import model_management as mm
                device = mm.get_torch_device()
                
                from uvr5_cli import Separator
                try:
                    model = Separator(
                        model_path=model_path,
                        device=device,
                        is_half="cuda" in str(device),
                        cache_dir=cache_dir,
                        agg=aggressiveness
                        )
                    primary, secondary, _ = model.run_inference(audio_path,format=format)
                    print(f"✅ RVC fallback completed successfully!")
                except RuntimeError as rvc_error:
                    if "incompatible with RVC fallback engine" in str(rvc_error):
                        print(f"🚫 {rvc_error}")
                        print(f"💡 Try using a different UVR model that's compatible with both engines")
                        raise RuntimeError(f"Model '{os.path.basename(model_path)}' is not supported in Python 3.13 environment. Both Audio-Separator and RVC fallback failed. Please try a different UVR model.")
                    else:
                        raise
            finally:
                if primary is not None and secondary is not None and use_cache:
                    print(f"💾 Caching results for faster future processing")
                    print(save_input_audio(primary_path,primary))
                    print(save_input_audio(secondary_path,secondary))

                if os.path.isfile(primary_path) and os.path.isfile(secondary_path) and use_cache:
                    primary = load_input_audio(primary_path)
                    secondary = load_input_audio(secondary_path)
        
        # Convert back to ComfyUI formats
        def to_audio_dict(audio_data, sample_rate):
            import torch
            if isinstance(audio_data, np.ndarray):
                if audio_data.ndim == 1:
                    waveform = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
                else:
                    waveform = torch.from_numpy(audio_data).float().unsqueeze(0)  # [1, channels, samples]
            else:
                waveform = torch.tensor(audio_data).float().unsqueeze(0).unsqueeze(0)
            
            return {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
        
        # Some models return vocals/instrumentals in opposite order
        model_name = filename.lower()  # Use original filename, not the reassigned model object
        
        # Models that typically return inverted outputs 
        if ("roformer" in model_name or "bs_roformer" in model_name or 
            ("karaoke" in model_name and "hp" in model_name) or
            "deecho" in model_name or "dereverb" in model_name):
            # Swap outputs for these models
            print(f"🔄 Model with inverted outputs detected - swapping (primary=instrumentals, secondary=vocals)")
            return (to_audio_dict(secondary[0], secondary[1]), to_audio_dict(primary[0], primary[1]))  # extracted=vocals, remaining=instrumentals
        else:
            # Standard order for most models
            return (to_audio_dict(primary[0], primary[1]), to_audio_dict(secondary[0], secondary[1]))  # extracted=vocals, remaining=instrumentals