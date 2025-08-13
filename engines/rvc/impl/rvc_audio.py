import base64
import io
import os
import zlib
import subprocess
from rvc_utils import get_hash, get_merge_func
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.interpolate import interp1d
from collections.abc import Mapping

MAX_INT16 = 32768
SUPPORTED_AUDIO = ["mp3","flac","wav"] # ogg breaks soundfile
OUTPUT_CHANNELS = ["mono", "stereo"]
AUTOTUNE_NOTES = np.array([
    65.41, 69.30, 73.42, 77.78, 82.41, 87.31,
    92.50, 98.00, 103.83, 110.00, 116.54, 123.47,
    130.81, 138.59, 146.83, 155.56, 164.81, 174.61,
    185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23,
    369.99, 392.00, 415.30, 440.00, 466.16, 493.88,
    523.25, 554.37, 587.33, 622.25, 659.25, 698.46,
    739.99, 783.99, 830.61, 880.00, 932.33, 987.77,
    1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91,
    1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53,
    2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83,
    2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07
])
SR_MAP = {"32k": 32000,"40k": 40000, "48k": 48000}

class AudioProcessor:
    def __init__(self, normalize=True, threshold_silence=True, dynamic_threshold=True, sample_size=16000, multiplier=2.0, fill_method="median", kernel_size=5, silence_threshold_db=-50, normalize_threshold_db=-1):
        self.normalize=normalize
        self.threshold_silence=threshold_silence
        self.dynamic_threshold=dynamic_threshold
        self.sample_size=sample_size
        self.multiplier=multiplier
        self.fill_method=fill_method
        self.kernel_size=kernel_size
        self.silence_threshold_db=silence_threshold_db
        self.normalize_threshold_db=normalize_threshold_db

    def __str__(self) -> str:
        values = [self.normalize, self.threshold_silence, self.dynamic_threshold]
        if self.normalize: values.append(self.normalize_threshold_db)
        if self.threshold_silence: values.append(self.silence_threshold_db)
        if self.dynamic_threshold: values.extend([self.sample_size, self.multiplier, self.fill_method, self.kernel_size])
        return get_hash(*values)

    def __call__(self, audio):
        
        samples,sr = get_audio(audio)
        
        if self.threshold_silence:
            from karafan.audio_utils import Silent
            samples = np.atleast_2d([samples])
            samples = Silent(samples, sample_rate=sr, threshold_dB=self.silence_threshold_db)
            samples = np.squeeze(samples, 0)
        if self.dynamic_threshold:
            samples = self.dynamic_thresholding(
            samples,
            multiplier=self.multiplier,
            sample_size=self.sample_size,
            method=self.fill_method,
            kernel_size=self.kernel_size)
        if self.normalize:
            from karafan.audio_utils import Normalize
            samples = Normalize(samples,threshold_dB=self.normalize_threshold_db)

        return samples, sr

    @staticmethod
    def dynamic_thresholding(samples, multiplier=2., sample_size=16000, method="median", kernel_size=5):
        # Calculate the local RMS (root mean square) to estimate the local amplitude
        local_rms = np.sqrt(uniform_filter1d(np.square(samples), size=int(sample_size)))
        
        # Set dynamic threshold as a multiple of the local RMS
        dynamic_threshold = multiplier * local_rms
        
        # Detect clicks where the absolute value exceeds the dynamic threshold
        clicks = np.abs(samples) > dynamic_threshold
        
        # Replace clicks using the specified method
        cleaned_samples = AudioProcessor.replace_clicks(samples, clicks, method=method, kernel_size=kernel_size)
        return cleaned_samples

    @staticmethod
    def replace_clicks(samples, clicks, method="median", kernel_size=5):
        if method == "median":
            # Replace detected clicks with the median value of surrounding samples
            cleaned_samples = samples.copy()
            cleaned_samples[clicks] = median_filter(samples, size=kernel_size)[clicks]
        elif method == "interpolation":
            # Replace detected clicks with interpolation
            cleaned_samples = samples.copy()
            
            # Indices of non-clicks (valid points for interpolation)
            non_click_indices = np.where(~clicks)[0]
            
            # Indices of clicks (points to be replaced)
            click_indices = np.where(clicks)[0]
            
            # Interpolation function
            interp_func = interp1d(non_click_indices, cleaned_samples[non_click_indices], kind='linear', bounds_error=False, fill_value="extrapolate")
            
            # Replace click samples with interpolated values
            cleaned_samples[click_indices] = interp_func(click_indices)
        else:
            raise ValueError("Method must be 'median' or 'interpolation'")
        
        return cleaned_samples

def get_audio(audio):
    if hasattr(audio,"__call__"): #VHS_AUDIO is a function
        audio = audio()
    if isinstance(audio,Mapping): #comfyui AUDIO
        # ComfyUI format: [batch, channels, samples] -> we want [channels, samples] or [samples] for mono
        waveform = audio["waveform"].squeeze(0).numpy()  # Remove batch dimension: [channels, samples]
        if waveform.shape[0] == 1:  # If mono, flatten to 1D
            waveform = waveform.squeeze(0)  # [samples]
        print(f"🎧 Processing {waveform.shape} audio at {audio['sample_rate']}Hz")
        return waveform, audio["sample_rate"]
    elif type(audio)==bytes: return bytes_to_audio(audio)
    else:
        print(f"{audio} is neither VHS_AUDIO or AUDIO format...")
        return audio

def load_audio(file, sr, **kwargs):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        # Direct subprocess call to ffmpeg instead of ffmpeg-python
        cmd = [
            "ffmpeg", "-nostdin", "-threads", "0", "-i", file,
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        out = result.stdout
    except Exception as error:
        raise RuntimeError(f"Failed to load audio: {error=} {file=}")

    return remix_audio((np.frombuffer(out, np.float32).flatten(), sr),**kwargs)

def remix_audio(input_audio,target_sr=None,norm=False,to_int16=False,resample=False,axis=0,merge_type=None,max_volume=.95,**kwargs):
    audio = np.array(input_audio[0],dtype="float32")
    if target_sr is None: target_sr=input_audio[1]

    # Debug info (uncomment if needed): shape={audio.shape}, max={audio.max()}, min={audio.min()}, mean={audio.mean()} sr={input_audio[1]}
    if resample or input_audio[1]!=target_sr:
        audio = librosa.resample(np.array(input_audio[0],dtype="float32"),orig_sr=input_audio[1],target_sr=target_sr,**kwargs)
    
    if audio.ndim>1:
        merge_func = get_merge_func(merge_type)
        audio=merge_func(audio,axis=axis)
    if norm: audio = librosa.util.normalize(audio,axis=axis)

    audio_max = np.abs(audio).max()/max_volume
    if audio_max > 1: audio = audio / audio_max
        
    if to_int16: audio = np.clip(audio * MAX_INT16, a_min=1-MAX_INT16, a_max=MAX_INT16-1).astype("int16")
    # Debug info (uncomment if needed): after remix: shape={audio.shape}, max={audio.max()}, min={audio.min()}, mean={audio.mean()}, sr={target_sr}

    return audio, target_sr

def load_input_audio(fname,sr=None,**kwargs):
    if sr is None: sr=44100
    audio, sr = load_audio(fname, sr, **kwargs)
    print(f"loading sound {fname=} {audio.shape=} {audio.max()=} {audio.min()=} {audio.dtype=} {sr=}")
    return audio, sr
   
def save_input_audio(fname,input_audio,sr=None,to_int16=False,to_stereo=False,max_volume=.99):
    print(f"saving sound to {fname}")
    if input_audio is None:
        print("Warning: input_audio is None, cannot save audio")
        return None
    os.makedirs(os.path.dirname(fname),exist_ok=True)
    audio=np.array(input_audio[0],dtype="float32")

    audio_max = np.abs(audio).max()/max_volume
    if audio_max > 1: audio = audio / audio_max
    if to_int16: audio = np.clip(audio * MAX_INT16, a_min=1-MAX_INT16, a_max=MAX_INT16-1)
    if to_stereo and audio.ndim<2: audio=np.stack([audio,audio],axis=-1)
    
    try:
        if audio.ndim>1 and audio.shape[0]<audio.shape[1]: audio=audio.T # soundfile expects data in frames x channels
        sf.write(fname, audio.astype("int16" if np.abs(audio).max()>1 else "float32"), sr if sr else input_audio[1])
        return f"File saved to ${fname}"
    except Exception as e:
        return f"failed to save audio: {e}"
    
def audio_to_bytes(audio,sr,target_sr=None,to_int16=False,to_stereo=False,format="WAV"):
    
    with io.BytesIO() as bytes_io:
        audio=np.array(audio,dtype="float32")

        if to_int16:
            audio_max = np.abs(audio).max()/.99
            if audio_max > 1: audio = audio / audio_max
            audio = np.clip(audio * MAX_INT16, a_min=-MAX_INT16+1, a_max=MAX_INT16-1)

        if to_stereo and audio.ndim<2: audio=np.stack([audio,audio],axis=-1)

        if audio.ndim>1 and audio.shape[0]<audio.shape[1]: audio=audio.T # soundfile expects data in frames x channels
        samplerate = sr if target_sr is None else target_sr
        sf.write(bytes_io, audio.astype("int16" if np.abs(audio).max()>1 else "float32"), samplerate=samplerate,format=format)
        bytes_io.seek(0)
        return bytes_io.read()

def bytes_to_audio(data: bytes,**kwargs):
    with io.BytesIO(data) as bytes_io:
        audio, sr = sf.read(bytes_io,**kwargs)
        if audio.ndim>1 and audio.shape[1]<audio.shape[0]: audio = audio.T # transpose to channels x frames
        return audio, sr

def bytes2audio(data: str):
    try:
        # Split the suffixed data by the colon
        dtype,data,shape,sr = data.split(":")

        # Get the data, the dtype, and the shape from the split data
        shape = tuple(map(int, shape.split(",")))
        sr=int(sr)

        # Decode the data using base64
        decoded_data = base64.b64decode(data)

        # Decompress the decoded data using zlib
        decompressed_data = zlib.decompress(decoded_data)

        # Convert the decompressed data to a numpy array with the given dtype
        arr = np.frombuffer(decompressed_data, dtype=dtype)

        # Reshape the array to the original shape
        arr = arr.reshape(shape)
        return arr, sr
    except Exception as e:
        print(e)
    return None

def audio2bytes(audio: np.array, sr: int):
    try:
        # Get the dtype, the shape, and the data of the array
        dtype = audio.dtype.name
        shape = audio.shape
        data = audio.tobytes()

        # Compress the data using zlib
        compressed_data = zlib.compress(data)

        # Encode the compressed data using base64
        encoded_data = base64.b64encode(compressed_data)

        # Add a suffix with the dtype and the shape to the encoded data
        suffixed_data = ":".join([dtype,encoded_data.decode(),",".join(map(str, shape)),str(sr)])
        return suffixed_data
    except Exception as e:
        print(e)
    return ""

def pad_audio(*audios,axis=0):
    maxlen = max(len(a) if a is not None else 0 for a in audios)
    if maxlen>0:
        stack = librosa.util.stack([librosa.util.fix_length(a,size=maxlen) for a in audios if a is not None],axis=axis)
        return stack
    else: return np.stack(audios,axis=axis)

def merge_audio(audio1,audio2,sr=40000,**kwargs):
    print(f"merging audio audio1={audio1[0].shape,audio1[1]} audio2={audio2[0].shape,audio2[1]} sr={sr}")
    if sr is None: sr=min(audio1[-1],audio2[-1])
    m1,_=remix_audio(audio1,target_sr=sr,axis=0,**kwargs)
    m2,_=remix_audio(audio2,target_sr=sr,axis=0,**kwargs)
    
    mixed = pad_audio(m1,m2,axis=0)

    return remix_audio((mixed,sr),axis=0,**kwargs)

def autotune_f0(f0, threshold=0.):
    # autotuned_f0 = []
    # for freq in f0:
        # closest_notes = [x for x in self.note_dict if abs(x - freq) == min(abs(n - freq) for n in self.note_dict)]
        # autotuned_f0.append(random.choice(closest_notes))
    # for note in self.note_dict:
    #     closest_notes = np.where((f0 - note)/note<.05,f0,note)
    print("autotuning f0 using note_dict...")

    autotuned_f0 = []
    # Loop through each value in array1
    for freq in f0:
        # Find the absolute difference between x and each value in array2
        diff = np.abs(AUTOTUNE_NOTES - freq)
        # Find the index of the minimum difference
        idx = np.argmin(diff)
        # Find the corresponding value in array2
        y = AUTOTUNE_NOTES[idx]
        # Check if the difference is less than threshold
        if diff[idx] < threshold:
            # Keep the value in array1
            autotuned_f0.append(freq)
        else:
            # Use the nearest value in array2
            autotuned_f0.append(y)
    # Return the result as a numpy array
    return np.array(autotuned_f0, dtype="float32")

def hz_to_mel(hz: "np.ndarray"):
    # 1127 * np.log(1 + f0_min / 700)
    return 2595 * np.log10(1 + hz / 700) # from wikipedia