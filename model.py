import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
import torch
import numpy as np
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT
import soundfile as sf
import io
from pydub import AudioSegment, silence
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os

class INF5Config(PretrainedConfig):
    model_type = "inf5"

    def __init__(self, ckpt_path: str = "checkpoints/model_best.pt", vocab_path: str = "checkpoints/vocab.txt", 
                 speed: float = 1.0, remove_sil: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.ckpt_path = ckpt_path
        self.vocab_path = vocab_path
        self.speed = speed
        self.remove_sil = remove_sil

class INF5Model(PreTrainedModel):
    config_class = INF5Config

    def __init__(self, config):
        super().__init__(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load vocoder
        self.vocoder = torch.compile(load_vocoder(vocoder_name="vocos", is_local=False, device=device))

        # Download and load model weights
        # safetensors_path = hf_hub_download(config.name_or_path, filename="model.safetensors")
        # print(f"Loading model weights from {safetensors_path} (safetensors)...")
        # state_dict = load_file(safetensors_path, device=str(device))

        # Download vocab.txt from HF Hub
        vocab_path = hf_hub_download(config.name_or_path, filename="checkpoints/vocab.txt")

        self.ema_model = torch.compile(load_model(
                DiT,
                dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                mel_spec_type="vocos",
                vocab_file=vocab_path,
                device=device
            )
        )

        # # Load state dict into model
        # self.ema_model.load_state_dict(state_dict, strict=False)

    
    def forward(self, text: str, ref_audio_path: str, ref_text: str):
        """
        Generate speech given a reference audio & text input.
        
        Args:
            text (str): The text to be synthesized.
            ref_audio_path (str): Path to the reference audio file.
            ref_text (str): The reference text.

        Returns:
            np.array: Generated waveform.
        """

        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio file {ref_audio_path} not found.")

        # Load reference audio & text
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)

        
        self.ema_model.to(self.device)
        self.vocoder.to(self.device)
        
        # Perform inference
        audio, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            text,
            self.ema_model,
            self.vocoder,
            mel_spec_type="vocos",
            speed=self.config.speed,
            device=self.device,
        )

        # Convert to pydub format and remove silence if needed
        buffer = io.BytesIO()
        sf.write(buffer, audio, samplerate=24000, format="WAV")
        buffer.seek(0)
        audio_segment = AudioSegment.from_file(buffer, format="wav")

        if self.config.remove_sil:
            non_silent_segs = silence.split_on_silence(
                audio_segment,
                min_silence_len=1000,
                silence_thresh=-50,
                keep_silence=500,
                seek_step=10,
            )
            non_silent_wave = sum(non_silent_segs, AudioSegment.silent(duration=0))
            audio_segment = non_silent_wave

        # Normalize loudness
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio_segment.dBFS
        audio_segment = audio_segment.apply_gain(change_in_dBFS)

        return np.array(audio_segment.get_array_of_samples())



if __name__ == '__main__':
    model = INF5Model(INF5Config(ckpt_path="checkpoints/model_best.pt", vocab_path="checkpoints/vocab.txt"))
    model.save_pretrained("INF5")
    model.config.save_pretrained("INF5")

    import numpy as np
    import soundfile as sf
    from transformers import AutoConfig, AutoModel
    
    AutoConfig.register("inf5", INF5Config)
    AutoModel.register(INF5Config, INF5Model)

    model = AutoModel.from_pretrained("INF5")
    audio = model("नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए.", 
                  ref_audio_path="prompts/PAN_F_HAPPY_00001.wav",
                  ref_text="भਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।")
    
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0  
    sf.write("samples/namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)

    from huggingface_hub import HfApi

    repo_id = "svp19/INF5"  # Change to your HF repo

    # Upload model directory to HF
    api = HfApi()
    api.upload_folder(
        folder_path="INF5",
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"Model pushed to https://huggingface.co/{repo_id} 🚀")

    print("Verify Upload")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(repo_id)
    print("Success")

