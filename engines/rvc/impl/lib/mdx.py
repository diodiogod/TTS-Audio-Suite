import json
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm
from lib.model_utils import get_hash

from lib.utils import gc_collect
try:
    import onnxruntime as ort
    # Verify essential attributes exist
    if not hasattr(ort, 'InferenceSession'):
        print(f"❌ ONNX Runtime installation is corrupted!")
        print(f"💡 Fix: pip uninstall onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu")
        print(f"🔄 Disabling ONNX models, fallback engines will be used")
        ort = None  # Disable ONNX functionality
        
except ImportError as e:
    print(f"❌ ONNX Runtime import failed: {e}")
    print(f"💡 Try: pip uninstall onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu")
    print(f"🔄 Disabling ONNX models, fallback engines will be used")
    ort = None

warnings.filterwarnings("ignore")
stem_naming = {'Vocals': 'Instrumental', 'Other': 'Instruments', 'Instrumental': 'Vocals', 'Drums': 'Drumless', 'Bass': 'Bassless'}


class MDXParams:
    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024, stem_name=None, compensation=1.000):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft//2
        self.chunk_size = hop * (self.dim_t - 1)
        self.device = device
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True,device=self.device)

        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins-self.dim_f, self.dim_t]).to(self.device)
        self.gen_size = self.chunk_size-2*self.trim

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True,return_complex=True)
        x=torch.view_as_real(x)
        x = x.permute([0,3,1,2])
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,self.dim_c,self.n_bins,self.dim_t])
        return x[:,:,:self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0],1,1,1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,2,self.n_bins,self.dim_t])
        x = x.permute([0,2,3,1])
        x=x.contiguous()
        x=torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1,2,self.chunk_size])

class MDXModel:
    DEFAULT_SR = 44100
    # Unit: seconds
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    DEFAULT_PROCESSOR = 0

    # def __init__(self, model_path: str, params: MDXModel, processor=DEFAULT_PROCESSOR,margin=DEFAULT_MARGIN_SIZE,chunks=15):
    def __init__(self, model_path: str, device="cpu",margin=DEFAULT_MARGIN_SIZE,chunks=15,denoise=False):
        
        # Check if ONNX Runtime is available
        if ort is None:
            raise RuntimeError("ONNX Runtime not available - MDX models cannot be used. Please fix ONNX Runtime installation.")

        # Set the device and the provider (CPU or CUDA)
        self.device = device
        
        # Try to use CUDA provider if available, otherwise fall back to CPU
        try:
            # Check if get_available_providers exists (newer ONNX Runtime versions)
            if hasattr(ort, 'get_available_providers'):
                available_providers = ort.get_available_providers()
            else:
                # Fallback for older ONNX Runtime versions
                try:
                    # Test if CUDA provider works by trying to create a session
                    test_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    available_providers = test_providers  # Assume both are available for now
                except:
                    available_providers = ['CPUExecutionProvider']
            
            if 'CUDAExecutionProvider' in available_providers and 'cuda' in str(device).lower():
                self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"✅ ONNX Runtime CUDA acceleration enabled")
            else:
                self.providers = ['CPUExecutionProvider']
                if 'cuda' in str(device).lower():
                    print(f"⚠️ CUDA requested but CUDAExecutionProvider not available in ONNX Runtime")
                    print(f"💡 Install onnxruntime-gpu for CUDA acceleration: pip install onnxruntime-gpu")
                print(f"🔄 Using CPU execution provider")
        except Exception as e:
            print(f"⚠️ Error detecting ONNX Runtime providers: {e}")
            self.providers = ['CPUExecutionProvider']
            print(f"🔄 Falling back to CPU execution provider")

        mp = self.get_params(model_path)
        self.params = MDXParams(
            self.device,
            dim_f=mp["mdx_dim_f_set"],
            dim_t=2 ** mp["mdx_dim_t_set"],
            n_fft=mp["mdx_n_fft_scale_set"],
            stem_name=mp["primary_stem"],
            compensation=mp["compensate"]
        )

        # Load the ONNX model using ONNX Runtime
        self.ort = ort.InferenceSession(model_path, providers=self.providers)
        print(f"onnx load done: {model_path} ({self.params})")

        # Use GPU tensors if CUDA is available and requested
        if 'CUDAExecutionProvider' in self.providers and 'cuda' in str(device).lower():
            self.run = lambda spec: self.ort.run(None, {'input': spec.detach().cpu().numpy()})[0]
        else:
            self.run = lambda spec: self.ort.run(None, {'input': spec.cpu().numpy()})[0]
        self.margin=margin
        self.chunks=chunks
        self.denoise=denoise
        self.mdx_batch_size=1

    def initialize_mix(self, mix, is_ckpt=False):
        # Ensure mix is stereo (2 channels) for both branches
        if mix.shape[0] == 1:
            mix = np.concatenate([mix, mix], axis=0)  # Duplicate mono to stereo
            
        if is_ckpt:
            pad = self.params.gen_size + self.params.trim - ((mix.shape[-1]) % self.params.gen_size)
            mixture = np.concatenate((np.zeros((2, self.params.trim), dtype='float32'),mix, np.zeros((2, pad), dtype='float32')), 1)
            num_chunks = mixture.shape[-1] // self.params.gen_size
            mix_waves = [mixture[:, i * self.params.gen_size: i * self.params.gen_size + self.params.chunk_size] for i in range(num_chunks)]
        else:
            mix_waves = []
            
            n_sample = mix.shape[1]
            pad = self.params.gen_size - n_sample % self.params.gen_size
            mix_p = np.concatenate((np.zeros((2,self.params.trim)), mix, np.zeros((2,pad)), np.zeros((2,self.params.trim))), 1)
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i:i+self.params.chunk_size])
                mix_waves.append(waves)
                i += self.params.gen_size
                
        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)

        return mix_waves, pad
    
    def demix_base(self, mix, is_ckpt=False, is_match_mix=False):
        chunked_sources = []
        for slice in tqdm(mix,"Processing audio:"):
            sources = []
            tar_waves_ = []
            mix_p = mix[slice]
            mix_waves, pad = self.initialize_mix(mix_p, is_ckpt=is_ckpt)
            mix_waves = mix_waves.split(self.mdx_batch_size)
            pad = mix_p.shape[-1] if is_ckpt else -pad
            with torch.no_grad():
                for mix_wave in mix_waves:
                    tar_waves = self.run_model(mix_wave, is_ckpt=is_ckpt, is_match_mix=is_match_mix)
                    tar_waves_.append(tar_waves)
                tar_waves_ = np.vstack(tar_waves_)[:, :, self.params.trim:-self.params.trim] if is_ckpt else tar_waves_
                tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :pad]
                start = 0 if slice == 0 else self.margin
                end = None if slice == list(mix.keys())[::-1][0] or self.margin == 0 else -self.margin
                sources.append(tar_waves[:,start:end]/self.params.compensation)
            chunked_sources.append(sources)
        sources = np.concatenate(chunked_sources, axis=-1)
        
        return sources

    def run_model(self, mix, is_ckpt=False, is_match_mix=False):
        
        spek = self.params.stft(mix.to(self.device))*self.params.compensation
        spek[:, :, :3, :] *= 0 

        if is_match_mix:
            spec_pred = spek.cpu().numpy()
        else:
            spec_pred = self.run(spek)*0.5-self.run(-spek)*0.5 if self.denoise else self.run(spek)

        if is_ckpt:
            return self.params.istft(spec_pred).cpu().detach().numpy()
        else: 
            return self.params.istft(torch.tensor(spec_pred).to(self.device)).cpu()[:,:,self.params.trim:-self.params.trim].transpose(0,1).reshape(2, -1).numpy()

    def __del__(self):
        try:
            if hasattr(self, 'ort'):
                del self.ort
            gc_collect()
        except:
            pass

    def get_params(self,model_path):
        model_hash = get_hash(model_path)
        config_path = os.path.join(os.path.dirname(model_path), 'model_data.json')
        
        try:
            with open(config_path) as infile:
                model_params = json.load(infile)
            params = model_params.get(model_hash)
            if params:
                return params
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Config file not found or invalid: {config_path}")
            
            # Try to download missing model_data.json file
            if isinstance(e, FileNotFoundError):
                print(f"🔄 Attempting to download missing model_data.json...")
                try:
                    # Try downloading from the official UVR repository first
                    download_url = "https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/models/MDX_Net_Models/model_data/model_data.json"
                    
                    import requests
                    response = requests.get(download_url)
                    if response.status_code == 200:
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)
                        with open(config_path, 'wb') as f:
                            f.write(response.content)
                        success = True
                    else:
                        success = False
                    
                    if success:
                        print(f"✅ Successfully downloaded model_data.json")
                        # Try loading again after download
                        with open(config_path) as infile:
                            model_params = json.load(infile)
                        params = model_params.get(model_hash)
                        if params:
                            return params
                    else:
                        print(f"❌ Failed to download model_data.json")
                except Exception as download_error:
                    print(f"❌ Download failed: {download_error}")
            
            print(f"📋 Using default parameters for MDX model")
        
        # Fallback default parameters for MDX models
        # UVR-MDX-NET-vocal_FT.onnx expects 3072 dimensions based on the ONNX error
        return {
            "compensate": 1.025,
            "mdx_dim_f_set": 3072,  # Fixed: was 2048, model expects 3072
            "mdx_dim_t_set": 8,
            "mdx_n_fft_scale_set": 6144,
            "primary_stem": "Vocals"
        }

    # @staticmethod
    # def segment(wave, combine=True, chunk_size=DEFAULT_CHUNK_SIZE, margin_size=DEFAULT_MARGIN_SIZE):
    #     """
    #     Segment or join segmented wave array

    #     Args:
    #         wave: (np.array) Wave array to be segmented or joined
    #         combine: (bool) If True, combines segmented wave array. If False, segments wave array.
    #         chunk_size: (int) Size of each segment (in samples)
    #         margin_size: (int) Size of margin between segments (in samples)

    #     Returns:
    #         numpy array: Segmented or joined wave array
    #     """

    #     if combine:
    #         processed_wave = None  # Initializing as None instead of [] for later numpy array concatenation
    #         for segment_count, segment in enumerate(wave):
    #             start = 0 if segment_count == 0 else margin_size
    #             end = None if segment_count == len(wave) - 1 else -margin_size
    #             if margin_size == 0:
    #                 end = None
    #             if processed_wave is None:  # Create array for first segment
    #                 processed_wave = segment[:, start:end]
    #             else:  # Concatenate to existing array for subsequent segments
    #                 processed_wave = np.concatenate((processed_wave, segment[:, start:end]), axis=-1)

    #     else:
    #         processed_wave = []
    #         sample_count = wave.shape[-1]

    #         if chunk_size <= 0 or chunk_size > sample_count:
    #             chunk_size = sample_count

    #         if margin_size > chunk_size:
    #             margin_size = chunk_size

    #         for segment_count, skip in enumerate(range(0, sample_count, chunk_size)):

    #             margin = 0 if segment_count == 0 else margin_size
    #             end = min(skip + chunk_size + margin_size, sample_count)
    #             start = skip - margin

    #             cut = wave[:, start:end].copy()
    #             processed_wave.append(cut)

    #             if end == sample_count:
    #                 break

    #     return processed_wave

    # def pad_wave(self, wave):
    #     """
    #     Pad the wave array to match the required chunk size

    #     Args:
    #         wave: (np.array) Wave array to be padded

    #     Returns:
    #         tuple: (padded_wave, pad, trim)
    #             - padded_wave: Padded wave array
    #             - pad: Number of samples that were padded
    #             - trim: Number of samples that were trimmed
    #     """
    #     n_sample = wave.shape[1]
    #     trim = self.params.n_fft // 2
    #     gen_size = self.params.chunk_size - 2 * trim
    #     pad = gen_size - n_sample % gen_size

    #     # Padded wave
    #     wave_p = np.concatenate((np.zeros((2, trim)), wave, np.zeros((2, pad)), np.zeros((2, trim))), 1)

    #     mix_waves = []
    #     for i in range(0, n_sample + pad, gen_size):
    #         waves = np.array(wave_p[:, i:i + self.params.chunk_size])
    #         mix_waves.append(waves)

    #     mix_waves = torch.tensor(mix_waves, dtype=torch.float32).cpu()

    #     return mix_waves, pad, trim

    # # def _process_wave(self, mix_waves, trim, pad, q: queue.Queue, _id: int):
    # def _process_wave(self, mix_waves, trim, pad):
    #     """
    #     Process each wave segment in a multi-threaded environment

    #     Args:
    #         mix_waves: (torch.Tensor) Wave segments to be processed
    #         trim: (int) Number of samples trimmed during padding
    #         pad: (int) Number of samples padded during padding
    #         q: (queue.Queue) Queue to hold the processed wave segments
    #         _id: (int) Identifier of the processed wave segment

    #     Returns:
    #         numpy array: Processed wave segment
    #     """
    #     mix_waves = mix_waves.split(1)
    #     with torch.no_grad():
    #         pw = []
    #         for mix_wave in mix_waves:
    #             # self.prog.update()
    #             spec = self.stft(mix_wave.to(self.device))
    #             processed_spec = torch.tensor(self.process(spec))
    #             processed_wav = self.istft(processed_spec)
    #             processed_wav = processed_wav[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).cpu().numpy()
    #             pw.append(processed_wav)

    #     processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
        
    #     return processed_signal

    # def process_wave(self, wave: np.array, mt_threads=1):
    #     """
    #     Process the wave array in a multi-threaded environment

    #     Args:
    #         wave: (np.array) Wave array to be processed
    #         mt_threads: (int) Number of threads to be used for processing

    #     Returns:
    #         numpy array: Processed wave array
    #     """
    #     # self.prog = tqdm(total=0)
    #     # chunk_size = wave.shape[-1] // (mt_threads if not self.chunks else (self.chunks*mt_threads))
    #     chunk_size = self.params.chunk_size
    #     margin = self.margin
    #     waves = self.segment(wave, False, chunk_size, margin)
        
    #     processed_batches = []
    #     for batch in tqdm(waves,desc="processing waves"):
    #         mix_waves, pad, trim = self.pad_wave(batch)
    #         processed_waves = self._process_wave(mix_waves, trim, pad)
    #         processed_batches.append(processed_waves)

    #     assert len(processed_batches) == len(waves), 'Incomplete processed batches, please reduce batch size!'
    #     segmented_mix = self.segment(processed_batches, True, chunk_size, margin)

    #     return segmented_mix