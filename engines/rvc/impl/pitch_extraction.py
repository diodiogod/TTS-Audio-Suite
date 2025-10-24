from functools import partial
from multiprocessing.pool import ThreadPool
import os
import numpy as np

# NumPy 2.x compatibility fix
if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.complex = complex
    np.bool = bool
from scipy import signal
import torch

from .lib.rmvpe import RMVPE
from .lib.audio import autotune_f0, pad_audio, hz_to_mel
from .lib import BASE_MODELS_DIR
from .lib.utils import gc_collect, get_merge_func, get_optimal_threads, get_optimal_torch_device

class FeatureExtractor:
    def __init__(self, tgt_sr, config, onnx=False):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        
        self.sr = 16000  # hubert sr
        self.window = 160
        self.f0_bins = 256
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.device = config.device
        self.onnx = onnx
        self.f0_method_dict = {
            "pm": self.get_pm,
            "harvest": self.get_harvest,
            "dio": self.get_dio,
            "rmvpe": self.get_rmvpe,
            "rmvpe_onnx": self.get_rmvpe,
            "rmvpe+": self.get_pitch_dependant_rmvpe,
            "crepe": self.get_f0_official_crepe_computation,
            "crepe-tiny": partial(self.get_f0_official_crepe_computation, model='model'),
            "mangio-crepe": self.get_f0_crepe_computation,
            "mangio-crepe-tiny": partial(self.get_f0_crepe_computation, model='model'),
            "fcpe": self.get_fcpe,
        }
        
    def __del__(self):
        if hasattr(self,"model_rmvpe"):
            del self.model_rmvpe
            gc_collect()

    def load_index(self, file_index):
        index = big_npy = None
        try:
            if type(file_index)==tuple: # loading file index to save time
                    print("Using preloaded file index.")
                    index,big_npy = file_index
            elif file_index == "":
                print("File index was empty.")
                index = None
                big_npy = None
            else:
                if os.path.isfile(file_index):
                    print(f"Attempting to load {file_index}....")
                else:
                    print(f"{file_index} was not found...")
                import faiss
                # Suppress faiss loading messages by redirecting stdout
                import sys
                from io import StringIO

                old_stdout = sys.stdout
                sys.stdout = StringIO()

                try:
                    index = faiss.read_index(file_index)
                finally:
                    sys.stdout = old_stdout

                # Only show success message, not the verbose loading details
                print(f"loaded index: {index}")
                big_npy = index.reconstruct_n(0, index.ntotal)
        except Exception as e:
            print(f"Could not open Faiss index file for reading. {e}")
        finally: return index, big_npy
    
    # Fork Feature: Compute f0 with the crepe method
    def get_f0_crepe_computation(
        self,
        x,
        f0_min,
        f0_max,
        *args,  # 512 before. Hop length changes the speed that the voice jumps to a different dramatic pitch. Lower hop lengths means more pitch accuracy but longer inference time.
        **kwargs,  # Either use crepe-tiny "tiny" or crepe "full". Default is full
    ):
        import torchcrepe
        x = x.astype(
            np.float32
        )  # fixes the F.conv2D exception. We needed to convert double to float.
        x /= np.quantile(np.abs(x), 0.999)
        torch_device = get_optimal_torch_device()
        audio = torch.from_numpy(x).to(torch_device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        hop_length = kwargs.get('crepe_hop_length', 160)
        model = kwargs.get('model', 'full') 
        print("Initiating prediction with a crepe_hop_length of: " + str(hop_length))
        pitch: torch.Tensor = torchcrepe.predict(
            audio,
            self.sr,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=torch_device,
            pad=True,
        )
        p_len = x.shape[0] // hop_length
        # Resize the pitch for final f0
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)
        return f0  # Resized f0
    
    def get_f0_official_crepe_computation(
        self,
        x,
        f0_min,
        f0_max,
        *args,
        **kwargs
    ):
        import torchcrepe
        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 512
        # Compute pitch using first gpu
        audio = torch.tensor(np.copy(x))[None].float()
        model = kwargs.get('model', 'full') 
        f0, pd = torchcrepe.predict(
            audio,
            self.sr,
            self.window,
            f0_min,
            f0_max,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        return f0

    def get_pm(self, x, *args, **kwargs):
        import parselmouth
        p_len = x.shape[0] // 160 + 1
        f0 = parselmouth.Sound(x, self.sr).to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=kwargs.get('f0_min'),
            pitch_ceiling=kwargs.get('f0_max'),
        ).selected_array["frequency"]
        
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            # print(pad_size, p_len - len(f0) - pad_size)
            f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return f0

    def get_harvest(self, x, *args, **kwargs):
        import pyworld
        f0_spectral = pyworld.harvest(
            x.astype(np.double),
            fs=self.sr,
            f0_ceil=kwargs.get('f0_max'),
            f0_floor=kwargs.get('f0_min'),
            frame_period=1000 * kwargs.get('hop_length', 160) / self.sr,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.sr)

    def get_dio(self, x, *args, **kwargs):
        import pyworld
        f0_spectral = pyworld.dio(
            x.astype(np.double),
            fs=self.sr,
            f0_ceil=kwargs.get('f0_max'),
            f0_floor=kwargs.get('f0_min'),
            frame_period=1000 * kwargs.get('hop_length', 160) / self.sr,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.sr)


    def get_rmvpe(self, x, *args, **kwargs):
        if not hasattr(self,"model_rmvpe"):
            # Check organized TTS/ location first, then legacy
            rmvpe_file = f"rmvpe.{'onnx' if self.onnx else 'pt'}"
            
            # Check ComfyUI models directory first, then local BASE_MODELS_DIR
            rmvpe_paths = []
            try:
                import folder_paths
                comfyui_models_dir = folder_paths.models_dir
                rmvpe_paths.extend([
                    os.path.join(comfyui_models_dir, "TTS", "RVC", rmvpe_file),  # ComfyUI organized
                    os.path.join(comfyui_models_dir, "TTS", rmvpe_file),  # ComfyUI TTS
                ])
            except ImportError:
                pass
            
            # Add local paths as fallback
            rmvpe_paths.extend([
                os.path.join(BASE_MODELS_DIR, "TTS", "RVC", rmvpe_file),  # Local organized
                os.path.join(BASE_MODELS_DIR, rmvpe_file)  # Local legacy
            ])
            
            rmvpe_path = None
            for path in rmvpe_paths:
                if os.path.exists(path):
                    rmvpe_path = path
                    break
            
            if not rmvpe_path:
                rmvpe_path = rmvpe_paths[0]  # Use organized path as default
            
            self.model_rmvpe = RMVPE(rmvpe_path, is_half=self.is_half, device=self.device, onnx=self.onnx)

        return self.model_rmvpe.infer_from_audio(x, thred=0.03)

    def get_pitch_dependant_rmvpe(self, x, f0_min=0, f0_max=40000, *args, **kwargs):
        if not hasattr(self,"model_rmvpe"):
            # Check organized TTS/ location first, then legacy
            rmvpe_file = f"rmvpe.{'onnx' if self.onnx else 'pt'}"
            
            # Check ComfyUI models directory first, then local BASE_MODELS_DIR
            rmvpe_paths = []
            try:
                import folder_paths
                comfyui_models_dir = folder_paths.models_dir
                rmvpe_paths.extend([
                    os.path.join(comfyui_models_dir, "TTS", "RVC", rmvpe_file),  # ComfyUI organized
                    os.path.join(comfyui_models_dir, "TTS", rmvpe_file),  # ComfyUI TTS
                ])
            except ImportError:
                pass
            
            # Add local paths as fallback
            rmvpe_paths.extend([
                os.path.join(BASE_MODELS_DIR, "TTS", "RVC", rmvpe_file),  # Local organized
                os.path.join(BASE_MODELS_DIR, rmvpe_file)  # Local legacy
            ])
            
            rmvpe_path = None
            for path in rmvpe_paths:
                if os.path.exists(path):
                    rmvpe_path = path
                    break
            
            if not rmvpe_path:
                rmvpe_path = rmvpe_paths[0]  # Use organized path as default
            
            self.model_rmvpe = RMVPE(rmvpe_path, is_half=self.is_half, device=self.device, onnx=self.onnx)

        return self.model_rmvpe.infer_from_audio_with_pitch(x, thred=0.03, f0_min=f0_min, f0_max=f0_max)

    def get_fcpe(self, x, f0_min=80, f0_max=880, *args, **kwargs):
        """
        Fast Context-based Pitch Estimation using torchfcpe
        Official FCPE implementation designed for real-time voice conversion
        """
        try:
            from torchfcpe import spawn_bundled_infer_model
            
            # Initialize FCPE model if not already loaded
            if not hasattr(self, "model_fcpe"):
                print("Loading FCPE model...")
                self.model_fcpe = spawn_bundled_infer_model(device=self.device)
            
            # Preprocess audio for FCPE
            audio = x.astype(np.float32)
            audio = audio / np.max(np.abs(audio))  # Normalize
            audio_length = len(audio)
            
            # Convert to tensor format expected by FCPE
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1)
            if str(self.device) != 'cpu':
                audio_tensor = audio_tensor.to(self.device)
            
            # Calculate target length for F0 output
            hop_size = 160  # Standard hop size for RVC
            f0_target_length = (audio_length // hop_size) + 1
            
            # Perform FCPE inference
            with torch.no_grad():
                f0 = self.model_fcpe.infer(
                    audio_tensor,
                    sr=self.sr,
                    decoder_mode='local_argmax',  # Recommended by torchfcpe
                    threshold=0.006,              # V/UV decision threshold
                    f0_min=f0_min,               # Minimum pitch (Hz)
                    f0_max=f0_max,               # Maximum pitch (Hz)
                    interp_uv=False,             # Don't interpolate unvoiced frames
                    output_interp_target_length=f0_target_length
                )
            
            # Convert back to numpy array
            if isinstance(f0, torch.Tensor):
                f0 = f0.squeeze().cpu().numpy()
            
            # Ensure correct length for RVC pipeline
            p_len = x.shape[0] // self.window + 1
            if len(f0) != p_len:
                # Interpolate to correct length if needed
                target_indices = np.linspace(0, len(f0) - 1, p_len)
                f0 = np.interp(target_indices, np.arange(len(f0)), f0)
            
            return f0
            
        except ImportError:
            print("⚠️ FCPE requires torchfcpe library (should be installed with requirements)")
            print("🔄 Falling back to RMVPE...")
            return self.get_rmvpe(x, *args, **kwargs)
        except Exception as e:
            print(f"⚠️ FCPE failed ({e}) - falling back to RMVPE")
            return self.get_rmvpe(x, *args, **kwargs)

    # Fork Feature: Acquire median hybrid f0 estimation calculation
    def get_f0_hybrid_computation(
        self,
        methods_list,
        merge_type,
        x,
        f0_min,
        f0_max,
        filter_radius,
        crepe_hop_length,
        time_step,
        **kwargs
    ):
        # Get various f0 methods from input to use in the computation stack
        params = {'x': x, 'f0_min': f0_min, 
          'f0_max': f0_max, 'time_step': time_step, 'filter_radius': filter_radius, 
          'crepe_hop_length': crepe_hop_length, 'model': "full"
        }
        
        f0_computation_stack = []

        print(f"Calculating f0 pitch estimations for methods: {methods_list}")
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        # Get f0 calculations for all methods specified

        def _get_f0(method,params):
            if method not in self.f0_method_dict:
                raise Exception(f"Method {method} not found.")
            f0 = self.f0_method_dict[method](**params)
            if method == 'harvest' and filter_radius > 2:
                f0 = signal.medfilt(f0, filter_radius)
                f0 = f0[1:]  # Get rid of first frame.
            return f0

        with ThreadPool(max(1,get_optimal_threads())) as pool:
            f0_computation_stack = pool.starmap(_get_f0,[(method,params) for method in methods_list])

        f0_computation_stack = pad_audio(*f0_computation_stack) # prevents uneven f0

        print(f"Calculating hybrid median f0 from the stack of: {methods_list} using {merge_type} merge")
        merge_func = get_merge_func(merge_type)
        f0_median_hybrid = merge_func(f0_computation_stack, axis=0)

        return f0_median_hybrid

    def get_f0(
        self,
        x,
        f0_up_key,
        f0_method,
        merge_type="median",
        filter_radius=3,
        crepe_hop_length=160,
        f0_autotune=False,
        rmvpe_onnx=False,
        inp_f0=None,
        f0_min=50,
        f0_max=1100,
        **kwargs
    ):
        time_step = self.window / self.sr * 1000
        f0_mel_min = hz_to_mel(f0_min)
        f0_mel_max = hz_to_mel(f0_max)
        params = {'x': x, 'f0_up_key': f0_up_key, 'f0_min': f0_min, 
          'f0_max': f0_max, 'time_step': time_step, 'filter_radius': filter_radius, 
          'crepe_hop_length': crepe_hop_length, 'model': "full", 'onnx': rmvpe_onnx
        }
        print(f"get_f0 {f0_method} unused params: {kwargs}")
        if hasattr(f0_method,"pop") and len(f0_method)==1: f0_method = f0_method.pop()
        if type(f0_method) == list:
            # Perform hybrid median pitch estimation
            f0 = self.get_f0_hybrid_computation(f0_method,merge_type,**params)
        else:
            f0 = self.f0_method_dict[f0_method](**params)

        if f0_autotune:
            f0 = autotune_f0(f0)

        f0 *= pow(2, f0_up_key / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        
        # f0bak = f0.copy()
        f0_mel = hz_to_mel(f0)
        f0_mel =  (f0_mel - f0_mel_min)*(self.f0_bins-2)/(f0_mel_max - f0_mel_min) + 1
        f0_mel = np.clip(f0_mel, a_min=1, a_max=self.f0_bins-1)
        f0_coarse = np.rint(f0_mel).astype(np.int16)

        return f0_coarse, f0  # 1-0