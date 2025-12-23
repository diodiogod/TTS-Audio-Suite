import io
import threading
import time
import os
import sys

import numpy as np
import torch
import torchaudio
import onnxruntime
import whisper

# CRITICAL: Add our bundled directory FIRST to sys.path to prevent conflicts with other custom nodes
_impl_dir = os.path.dirname(os.path.abspath(__file__))
# Remove any existing paths that might conflict
sys.path = [p for p in sys.path if 'Step_Audio_EditX_TTS' not in p and 'Step-Audio-EditX' not in p or p == _impl_dir]
# Insert our bundled directory at the very beginning
if _impl_dir in sys.path:
    sys.path.remove(_impl_dir)
sys.path.insert(0, _impl_dir)

from funasr_detach import AutoModel
from model_loader import model_loader, ModelSource

# Import from local utils.py (not TTS Suite's utils package)
import importlib.util
_utils_spec = importlib.util.spec_from_file_location("step_audio_utils", os.path.join(_impl_dir, "utils.py"))
_step_audio_utils = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_step_audio_utils)
resample_audio = _step_audio_utils.resample_audio
energy_norm_fn = _step_audio_utils.energy_norm_fn
trim_silence = _step_audio_utils.trim_silence


class StepAudioTokenizer:
    def __init__(
        self,
        encoder_path,
        model_source=ModelSource.AUTO,
        funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
    ):
        """
        Initialize StepAudioTokenizer

        Args:
            encoder_path: Encoder path
            model_source: Model source (auto/local/modelscope/huggingface)
            funasr_model_id: FunASR model ID or path
        """
        # Load FunASR model for VQ02 encoding (should be downloaded by engine loader)
        base_path = os.path.dirname(encoder_path)  # Get parent of Step-Audio-EditX
        local_funasr_path = os.path.join(base_path, "FunASR-Paraformer")

        if not os.path.exists(local_funasr_path):
            raise FileNotFoundError(
                f"FunASR model not found at {local_funasr_path}. "
                f"This should have been auto-downloaded by the engine loader."
            )

        # Load FunASR model (silent - errors will be raised if fails)
        self.funasr_model = AutoModel(
            model=local_funasr_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            disable_log=True  # Suppress verbose registration tables
        )

        # Tokenizer files are in encoder_path (Step-Audio-EditX directory)
        kms_path = os.path.join(encoder_path, "linguistic_tokenizer.npy")
        cosy_tokenizer_path = os.path.join(encoder_path, "speech_tokenizer_v1.onnx")

        if not os.path.exists(kms_path):
            raise FileNotFoundError(f"KMS file not found: {kms_path}")
        if not os.path.exists(cosy_tokenizer_path):
            raise FileNotFoundError(f"Cosy tokenizer file not found: {cosy_tokenizer_path}")

        self.kms = torch.tensor(np.load(kms_path))

        # Auto-detect available ONNX providers (prefer CUDA if available)
        available_providers = onnxruntime.get_available_providers()
        providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]

        session_option = onnxruntime.SessionOptions()
        session_option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_option.intra_op_num_threads = 1
        session_option.log_severity_level = 3  # Suppress performance warnings (3=Error, 4=Fatal)
        self.ort_session = onnxruntime.InferenceSession(
            cosy_tokenizer_path, sess_options=session_option, providers=providers
        )
        self.chunk_size = [0, 4, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1

        self.vq02_sessions = {}
        self.vq02_lock = threading.Lock()
        self.vq06_lock = threading.Lock()

    def __call__(self, audio, sr):
        _, vq02, vq06 = self.wav2token(audio, sr, False)
        text = self.merge_vq0206_to_token_str(vq02, vq06)
        return text

    def preprocess_wav(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = resample_audio(audio, sample_rate, 16000)
        if energy_norm:
            audio = energy_norm_fn(audio)

        if enable_trim:
            audio = audio.cpu().numpy().squeeze(0)
            audio = trim_silence(audio, 16000)
            audio = torch.from_numpy(audio)
            audio = audio.unsqueeze(0)
        return audio

    def wav2token(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = self.preprocess_wav(
            audio, sample_rate, enable_trim=enable_trim, energy_norm=energy_norm
        )

        vq02_ori = self.get_vq02_code(audio)
        vq02 = [int(x) + 65536 for x in vq02_ori]
        vq06_ori = self.get_vq06_code(audio)
        vq06 = [int(x) + 65536 + 1024 for x in vq06_ori]

        chunk = 1
        chunk_nums = min(len(vq06) // (3 * chunk), len(vq02) // (2 * chunk))
        speech_tokens = []
        for idx in range(chunk_nums):
            speech_tokens += vq02[idx * chunk * 2 : (idx + 1) * chunk * 2]
            speech_tokens += vq06[idx * chunk * 3 : (idx + 1) * chunk * 3]
        return speech_tokens, vq02_ori, vq06_ori

    def get_vq02_code(self, audio, session_id=None, is_final=True):
        # 📦 BUNDLED CODE PATCH: Use temporary file instead of BytesIO for Windows compatibility
        # Issue: torchcodec on Windows fails with BytesIO for WAV format
        # Solution: Save to temp file, then read into BytesIO for FunASR
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            torchaudio.save(tmp_path, audio, 16000, format="wav")

            # Read the WAV file into BytesIO for FunASR processing
            with open(tmp_path, 'rb') as f:
                _tmp_wav = io.BytesIO(f.read())
            _tmp_wav.seek(0)
        finally:
            # Clean up temp file
            import os
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass  # Ignore cleanup errors

        with self.vq02_lock:
            cache = {}
            if session_id in self.vq02_sessions:
                cache = self.vq02_sessions[session_id].get("cache", {})

            res, new_cache = self.funasr_model.infer_encoder(
                input=[_tmp_wav],
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
                device=0,
                is_final=is_final,
                cache=cache,
            )
            c_list = []
            for j, res_ in enumerate(res):
                feat = res_["enc_out"]
                if len(feat) > 0:
                    c_list = self.dump_label([feat], self.kms)[0]

            if is_final:
                if session_id in self.vq02_sessions:
                    self.vq02_sessions.pop(session_id)
            else:
                if isinstance(session_id, str) and len(session_id) > 0:
                    self.vq02_sessions[session_id] = {
                        "cache": new_cache,
                        "update_time": time.time(),
                    }

            return c_list

    def get_vq06_code(self, audio):

        def split_audio(audio, chunk_duration=480000):
            start = 0
            chunks = []
            while start < len(audio):
                end = min(start + chunk_duration, len(audio))
                chunk = audio[start:end]
                if len(chunk) < 480:
                    pass
                else:
                    chunks.append(chunk)
                start = end
            return chunks

        with self.vq06_lock:
            audio = audio.squeeze(0)
            chunk_audios = split_audio(audio, chunk_duration=30 * 16000)  # Maximum support 30s
            speech_tokens = []
            for chunk in chunk_audios:
                duration = round(chunk.shape[0] / 16000, 2)
                feat = whisper.log_mel_spectrogram(chunk, n_mels=128)
                feat = feat.unsqueeze(0)
                feat_len = np.array([feat.shape[2]], dtype=np.int32)
                chunk_token = (
                    self.ort_session.run(
                        None,
                        {
                            self.ort_session.get_inputs()[0]
                            .name: feat.detach()
                            .cpu()
                            .numpy(),
                            self.ort_session.get_inputs()[1].name: feat_len,
                        },
                    )[0]
                    .flatten()
                    .tolist()
                )
                assert abs(len(chunk_token) - duration * 25) <= 2
                speech_tokens += chunk_token

            return speech_tokens

    def kmean_cluster(self, samples, means):
        dists = torch.cdist(samples, means)
        indices = dists.argmin(dim=1).cpu().numpy()
        return indices.tolist()

    def dump_label(self, samples, mean):
        dims = samples[0].shape[-1]
        x_lens = [x.shape[1] for x in samples]
        total_len = sum(x_lens)
        x_sel = torch.FloatTensor(1, total_len, dims)
        start_len = 0
        for sample in samples:
            sample_len = sample.shape[1]
            end_len = start_len + sample_len
            x_sel[:, start_len:end_len] = sample
            start_len = end_len
        dense_x = x_sel.squeeze(0)
        indices = self.kmean_cluster(dense_x, mean)
        indices_list = []
        start_len = 0
        for x_len in x_lens:
            end_len = start_len + end_len
            indices_list.append(indices[start_len:end_len])
        return indices_list

    def merge_vq0206_to_token_str(self, vq02, vq06):
        _vq06 = [1024 + x for x in vq06]
        result = []
        i = 0
        j = 0
        while i < len(vq02) - 1 and j < len(_vq06) - 2:
            sublist = vq02[i : i + 2] + _vq06[j : j + 3]
            result.extend(sublist)
            i += 2
            j += 3
        return "".join([f"<audio_{x}>" for x in result])
