import argparse
import os, torch, warnings

from lib.separators import MDXNet, UVR5Base, UVR5New
from lib import karafan
from lib.audio import load_input_audio, pad_audio, remix_audio, save_input_audio
from lib.utils import gc_collect, get_optimal_threads, get_merge_func

# Define paths
BASE_CACHE_DIR = "/tmp"

CACHED_SONGS_DIR = os.path.join(BASE_CACHE_DIR,"songs")

warnings.filterwarnings("ignore")
import numpy as np

class Separator:
    def __init__(self, model_path, use_cache=False, device="cpu", cache_dir=None, **kwargs):
        dereverb = "reverb" in model_path.lower()
        deecho = "echo"  in model_path.lower()
        bve = "BVE" in model_path.lower()
        denoise = dereverb or deecho or bve

        # Proper architecture detection based on actual model types
        model_name = os.path.basename(model_path).lower()
        
        if ("mdx" in model_path and "uvr-mdx-net" in model_name) or model_name.endswith('.onnx'):
            # MDX-Net models (like UVR-MDX-NET-vocal_FT.onnx)
            self.model = MDXNet(model_path=model_path,denoise=denoise,device=device,**kwargs)
        elif "scnet" in model_name:
            # SCNet models - new SOTA architecture from ZFTurbo
            # Remove aggressiveness parameter as SCNet doesn't use it
            kwargs_no_agg = {k: v for k, v in kwargs.items() if k != 'agg'}
            from lib.scnet import SCNetSeparator
            self.model = SCNetSeparator(model_path=model_path,device=device,**kwargs_no_agg)
            print(f"⚠️ Using SCNet EXPERIMENTAL architecture for: {os.path.basename(model_path)}")
            print(f"⚠️ WARNING: SCNet may produce audio artifacts (buzzing) - quality issues under investigation")
        elif "mdx23c" in model_name:
            # MDX23C models - new SOTA architecture from ZFTurbo
            # Remove aggressiveness parameter as MDX23C doesn't use it
            print(f"⚠️ MDX23C model detected: {os.path.basename(model_path)}")
            print(f"❌ ERROR: MDX23C has encoder-decoder tensor dimension mismatches")
            print(f"🔬 TECHNICAL: Skip connection concatenation fails due to off-by-one STFT frame alignment")
            print(f"💡 WORKAROUND: Use UVR-MDX-NET-vocal_FT.onnx or MelBand RoFormer models instead")
            # Fall through to default - will fail but with clear error message
        elif "roformer" in model_name or "bs_roformer" in model_name:
            # BS-RoFormer models - these are MDXC architecture, not VR
            # RoFormer models still need agg parameter but ignore it internally
            self.model = UVR5Base(model_path=model_path,device=device,**kwargs)
            print(f"🎵 Using RoFormer architecture for: {os.path.basename(model_path)}")
        elif any(x in model_name for x in ['hp5', 'denoise', 'deecho', 'dereverb', 'bve', 'karaoke']):
            # True VR Architecture models - these DO use aggressiveness
            self.model = UVR5New(model_path=model_path,device=device,dereverb=dereverb,**kwargs) if denoise else UVR5Base(model_path=model_path,device=device,**kwargs)
        else:
            # Default fallback for unknown models - assume VR architecture for backward compatibility
            self.model = UVR5New(model_path=model_path,device=device,dereverb=dereverb,**kwargs) if denoise else UVR5Base(model_path=model_path,device=device,**kwargs)
            
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.model_path = model_path
        self.args = kwargs
    
    # cleanup memory
    def __del__(self):
        gc_collect()

    def run_inference(self, audio_path, format="mp3"):
        song_name = get_filename(os.path.basename(self.model_path).split(".")[0],**self.args) + f".{format}"
        
        # handles loading of previous processed data
        music_dir = os.path.join(
            os.path.dirname(audio_path) if self.cache_dir is None else self.cache_dir,
            os.path.basename(audio_path).split(".")[0])
        vocals_path = os.path.join(music_dir,".vocals")
        instrumental_path = os.path.join(music_dir,".instrumental")
        vocals_file = os.path.join(vocals_path,song_name)
        instrumental_file = os.path.join(instrumental_path,song_name)

        if os.path.isfile(instrumental_file) and os.path.isfile(vocals_file):
            vocals = load_input_audio(vocals_file,mono=True)
            instrumental = load_input_audio(instrumental_file,mono=True)
            input_audio = load_input_audio(audio_path,mono=True)
            return vocals, instrumental, input_audio
        
        return_dict = self.model.run_inference(audio_path)
        
        # Handle different return formats - SCNet returns tuple, others return dict
        if isinstance(return_dict, tuple):
            # SCNet returns (vocals, instrumentals, input_audio)
            vocals, instrumental, input_audio = return_dict
        else:
            # Other models return dict
            instrumental = return_dict["instrumentals"]
            vocals = return_dict["vocals"] 
            input_audio = return_dict["input_audio"]
        

        if self.use_cache:
            os.makedirs(vocals_path,exist_ok=True)
            os.makedirs(instrumental_path,exist_ok=True)
            save_input_audio(vocals_file,vocals,to_int16=True)
            save_input_audio(instrumental_file,instrumental,to_int16=True)

        return vocals, instrumental, input_audio

def get_filename(*args,**kwargs):
    name = "_".join([str(arg) for arg in args]+[f"{k}={v}" for k,v in kwargs.items()])
    return name

def __run_inference_worker(arg):
    (model_path,audio_path,agg,device,use_cache,cache_dir,num_threads,format) = arg
    if "karafan" in model_path:
        vocals, instrumental, input_audio = karafan.inference.Process(audio_path,cache_dir=cache_dir,use_cache=use_cache,format=format)
    else:
        model = Separator(
                agg=agg,
                model_path=model_path,
                device=device,
                is_half="cuda" in str(device),
                use_cache=use_cache,
                cache_dir=cache_dir,
                num_threads = num_threads
                )
        vocals, instrumental, input_audio = model.run_inference(audio_path,format)
        del model
        gc_collect()

    return vocals, instrumental, input_audio

def split_audio(uvr_models,audio_path,preprocess_models=[],postprocess_models=[],device="cuda",agg=10,use_cache=False,merge_type="mean",format="mp3",**kwargs):
    print(f"unused kwargs={kwargs}")
    merge_func = get_merge_func(merge_type)
    num_threads = max(get_optimal_threads(-1),1)
    song_name = os.path.basename(audio_path).split(".")[0]
    cache_dir = os.path.join(CACHED_SONGS_DIR,song_name)

    # preprocess input song to split reverb
    if len(preprocess_models):
        output_name = get_filename(*[os.path.basename(name).split(".")[0] for name in preprocess_models],agg=agg) + f".{format}"
        preprocessed_file = os.path.join(cache_dir,"preprocessing",output_name)
        
        # read from cache
        if os.path.isfile(preprocessed_file): input_audio = load_input_audio(preprocessed_file,mono=True)
        else: # preprocess audio
            for i,preprocess_model in enumerate(preprocess_models):
                output_name = get_filename(i,os.path.basename(preprocess_model).split(".")[0],agg=agg) + f".{format}"
                intermediary_file = os.path.join(cache_dir,"preprocessing",output_name)
                if os.path.isfile(intermediary_file):
                    # if i==len(preprocess_model)-1: #last model
                    instrumental = input_audio = load_input_audio(intermediary_file, mono=True)
                else:
                    args = (preprocess_model,audio_path,agg,device,False,CACHED_SONGS_DIR if i==0 else None,num_threads,format)
                    _, instrumental, input_audio = __run_inference_worker(args)
                    save_input_audio(intermediary_file,instrumental,to_int16=True)
                audio_path = intermediary_file

            save_input_audio(preprocessed_file,instrumental,to_int16=True)
        audio_path = preprocessed_file
    else:
        input_audio = load_input_audio(audio_path,mono=True)
        
    # apply vocal separation
    wav_instrument = []
    wav_vocals = []

    for model_path in uvr_models:
        args = (model_path,audio_path,agg,device,use_cache,cache_dir,num_threads,format)
        print(f"processing... {args=}")
        vocals, instrumental, _ = __run_inference_worker(args)
        wav_vocals.append(vocals[0])
        wav_instrument.append(instrumental[0])
    wav_instrument = np.nanmedian(pad_audio(*wav_instrument,axis=0),axis=0)
    wav_vocals = merge_func(pad_audio(*wav_vocals,axis=0),axis=0)

    # postprocess vocals to reduce reverb
    if len(postprocess_models):
        vocals_name = get_filename("vocals",*[os.path.basename(name).split(".")[0] for name in uvr_models],agg=agg) + f".{format}"
        vocals_file = os.path.join(cache_dir,"postprocessing",vocals_name)
        if not os.path.isfile(vocals_file): save_input_audio(vocals_file,(wav_vocals,vocals[-1]),to_int16=True)
        print("postprocessing...")        
        for i,postprocess_model in enumerate(postprocess_models):
            output_name = get_filename(i,os.path.basename(postprocess_model).split(".")[0],agg=agg) + f".{format}"
            intermediary_file = os.path.join(cache_dir,"postprocessing",output_name)
            if not os.path.isfile(intermediary_file):
                args = (postprocess_model,vocals_file,agg,device,False,None,num_threads,format)
                _, processed_audio, _ = __run_inference_worker(args)
                output_name = get_filename(i,os.path.basename(postprocess_model).split(".")[0],agg=agg) + f".{format}"
                save_input_audio(intermediary_file,processed_audio,to_int16=True)
                wav_vocals, _ = processed_audio
            vocals_file = intermediary_file

    instrumental = remix_audio((wav_instrument,instrumental[-1]),to_int16=True)
    vocals = remix_audio((wav_vocals,vocals[-1]),to_int16=True)

    return vocals, instrumental, input_audio

def main(): #uvr5_models,audio_path,device="cuda",agg=10,use_cache=False
    
    parser = argparse.ArgumentParser(description="processes audio to split vocal stems and reduce reverb/echo")
    
    parser.add_argument("uvr5_models", type=str, nargs="+", help="Path to models to use for processing")
    parser.add_argument(
        "-i", "--audio_path", type=str, help="path to audio file to process", required=True
    )
    parser.add_argument(
        "-p", "--preprocess_model", type=str, help="preprocessing model to improve audio", default=None
    )
    parser.add_argument(
        "-a", "--agg", type=int, default=10, help="aggressiveness score for processing (0-20)"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", choices=["cpu","cuda"], help="perform calculations on [cpu] or [cuda]"
    )
    parser.add_argument(
        "-m", "--merge_type", type=str, default="median", choices=["mean","median"], help="how to combine processed audio"
    )
    parser.add_argument(
        "-c", "--use_cache", type=bool, action="store_true", default=False, help="caches the results so next run is faster"
    )
    args = parser.parse_args()
    return split_audio(**vars(args))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()