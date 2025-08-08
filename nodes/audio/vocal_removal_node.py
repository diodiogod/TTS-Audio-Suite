import os
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
from rvc_downloader import KARAFAN_MODELS, MDX_MODELS, RVC_DOWNLOAD_LINK, VR_MODELS, download_file

# Define paths
BASE_CACHE_DIR = folder_paths.get_temp_directory()
BASE_MODELS_DIR = folder_paths.models_dir

temp_path = folder_paths.get_temp_directory()
cache_dir = os.path.join(BASE_CACHE_DIR,"uvr")
device = get_optimal_torch_device()
is_half = True

class VocalRemovalNode:
    
    @classmethod
    def NAME(cls):
        return "🤐 Noise or Vocal Removal"
 
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):

        model_list = MDX_MODELS + VR_MODELS + KARAFAN_MODELS + get_filenames(root=BASE_MODELS_DIR,format_func=lambda x: f"{os.path.basename(os.path.dirname(x))}/{os.path.basename(x)}",name_filters=["UVR","MDX","karafan"])
        model_list = list(set(model_list)) # dedupe
        
        # Filter out non-model files (JSON configs, etc.)
        model_extensions = ['.pth', '.ckpt', '.onnx', '.pt', '.safetensors']
        model_list = [model for model in model_list if any(model.lower().endswith(ext) for ext in model_extensions)]

        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio for vocal/instrumental separation. Standard ComfyUI AUDIO format."
                }),
                "model": (model_list,{
                    "default": "UVR/HP5-vocals+instrumentals.pth",
                    "tooltip": """🎵 AI AUDIO SEPARATION & PROCESSING

🏆 TOP MODELS (2024-2025):
• model_bs_roformer_ep_317_sdr_12.9755.ckpt - ⭐ BEST OVERALL (Transformer SOTA)
• MDX23C-8KFFT-InstVoc_HQ.ckpt - Highest quality, minimal artifacts
• UVR-MDX-NET-vocal_FT.onnx - Professional vocal extraction

🎯 QUICK START:
• 🎤 Karaoke: model_bs_roformer + aggressive (12-15)
• 🎵 Vocal Extraction: UVR-MDX-NET-vocal_FT + balanced (8-12)
• 🔧 Denoising: UVR-DeNoise + gentle (5-8)
• 🏠 Beginner: HP5-vocals+instrumentals + moderate (10)

⚠️ SPECIAL MODELS:
• UVR-DeNoise - NOISE REMOVAL: "remaining" = clean audio ✅
• UVR-DeEcho-DeReverb - ECHO REMOVAL: "remaining" = dry audio ✅

📖 Complete guide with all models & workflows: docs/VOCAL_REMOVAL_GUIDE.md"""
                }),
            },
            "optional": {
                "use_cache": ("BOOLEAN",{
                    "default": True,
                    "tooltip": """🚀 CACHING SYSTEM

Enables intelligent caching of separation results for faster processing:
• ✅ ON (Recommended): Saves results to disk, dramatically speeds up repeated processing of same audio/model combinations
• ❌ OFF: Always processes from scratch, uses more time but ensures fresh results

💡 Cache includes model, aggression, format, and audio content in hash
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

⚠️ NOTE: Only affects VR models (.pth files). MDX (.onnx) and Karafan (.ckpt) models ignore this setting.
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

    CATEGORY = "🎵 TTS Audio Suite/Audio"

    def split(self, audio, model, use_cache=True, aggressiveness=10, format='flac'):
        filename = os.path.basename(model)
        subfolder = os.path.dirname(model)
        model_path = os.path.join(BASE_MODELS_DIR,subfolder,filename)
        if not os.path.isfile(model_path):
            download_link = f"{RVC_DOWNLOAD_LINK}{model}"
            params = model_path, download_link
            if download_file(params): print(f"successfully downloaded: {model_path}")
        
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
                if "karafan" in model_path: # try karafan implementation
                    print(f"🔧 Using Karafan separation engine")
                    primary, secondary, _ = karafan.inference.Process(audio_path,cache_dir=temp_path,format=format)
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
                print(f"⚠️ Primary engine failed (model not in supported list), switching to RVC fallback engine...")
                print(f"💡 This is normal - downloading and using model with RVC implementation")
                
                from uvr5_cli import Separator
                model = Separator(
                    model_path=model_path,
                    device=device,
                    is_half="cuda" in str(device),
                    cache_dir=cache_dir,
                    agg=aggressiveness
                    )
                primary, secondary, _ = model.run_inference(audio_path,format=format)
                print(f"✅ RVC fallback completed successfully!")
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