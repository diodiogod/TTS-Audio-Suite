"""
Dataset preparation for RVC voice model training.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import traceback
import zipfile
from pathlib import Path
from threading import Lock, Thread
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import folder_paths
import numpy as np
import torch
from scipy.io import wavfile

from engines.rvc.hubert_downloader import ensure_hubert_model
from engines.rvc.impl.lib.audio import SR_MAP, SUPPORTED_AUDIO, hz_to_mel, load_input_audio, remix_audio
from engines.rvc.impl.lib.model_utils import load_hubert
from engines.rvc.impl.lib.slicer2 import Slicer
from engines.rvc.impl.lib.utils import gc_collect
from engines.rvc.impl.pitch_extraction import FeatureExtractor
from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.processing import AudioProcessingUtils


MUTE_DATASET_DIR = (
    Path(__file__).resolve().parent / "assets" / "mute"
)


class ProgressReporter:
    def __init__(self, label: str, total: int, step: int = 25):
        self.label = label
        self.total = max(int(total), 0)
        self.step = max(int(step), 1)
        self.current = 0
        self.skipped = 0
        self._lock = Lock()

    def advance(self, skipped: bool = False) -> None:
        with self._lock:
            self.current += 1
            if skipped:
                self.skipped += 1

            should_log = (
                self.current == 1
                or self.current == self.total
                or (self.current % self.step) == 0
            )
            if should_log:
                suffix = f" ({self.skipped} cached)" if self.skipped else ""
                print(f"{self.label}: {self.current}/{self.total}{suffix}")


def _slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    safe = safe.strip("_")
    return safe or "rvc_model"


def _resolve_source_path(dataset_source: str) -> str:
    if not dataset_source or not str(dataset_source).strip():
        raise ValueError("dataset_source is required")

    raw_path = os.path.expanduser(dataset_source)
    candidates = [raw_path]
    input_dir = folder_paths.get_input_directory()
    candidates.append(os.path.join(input_dir, raw_path))
    candidates.append(os.path.join(input_dir, "datasets", raw_path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(f"Dataset source not found: {dataset_source}")


def _iter_audio_files(root: str) -> Iterable[str]:
    allowed = {f".{ext.lower()}" for ext in SUPPORTED_AUDIO}
    for current_root, _, files in os.walk(root):
        for filename in sorted(files):
            if os.path.splitext(filename)[1].lower() in allowed:
                yield os.path.join(current_root, filename)


def _flatten_audio_source(source_path: str, destination_dir: str) -> int:
    os.makedirs(destination_dir, exist_ok=True)

    if os.path.isdir(source_path):
        files = list(_iter_audio_files(source_path))
        if not files:
            raise RuntimeError(f"No audio files found in directory: {source_path}")
        for idx, file_path in enumerate(files):
            _, ext = os.path.splitext(file_path)
            target = os.path.join(destination_dir, f"{idx:05d}_{os.path.basename(file_path)}")
            shutil.copy2(file_path, target)
        return len(files)

    if zipfile.is_zipfile(source_path):
        count = 0
        allowed = {f".{ext.lower()}" for ext in SUPPORTED_AUDIO}
        with zipfile.ZipFile(source_path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                basename = os.path.basename(member.filename)
                ext = os.path.splitext(basename)[1].lower()
                if not basename or ext not in allowed:
                    continue
                target = os.path.join(destination_dir, f"{count:05d}_{basename}")
                with archive.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
        if count == 0:
            raise RuntimeError(f"No supported audio files found in zip archive: {source_path}")
        return count

    ext = os.path.splitext(source_path)[1].lower()
    if ext in {f".{audio_ext}" for audio_ext in SUPPORTED_AUDIO}:
        target = os.path.join(destination_dir, os.path.basename(source_path))
        shutil.copy2(source_path, target)
        return 1

    raise ValueError(f"Unsupported dataset source: {source_path}")


def _fingerprint_source_path(source_path: str) -> str:
    if os.path.isdir(source_path):
        digest = hashlib.md5()
        files = list(_iter_audio_files(source_path))
        if not files:
            raise RuntimeError(f"No audio files found in directory: {source_path}")
        for file_path in files:
            relative_path = os.path.relpath(file_path, source_path).replace("\\", "/")
            stat = os.stat(file_path)
            digest.update(f"{relative_path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
        return f"dir:{digest.hexdigest()}"

    if zipfile.is_zipfile(source_path):
        digest = hashlib.md5()
        allowed = {f".{ext.lower()}" for ext in SUPPORTED_AUDIO}
        with zipfile.ZipFile(source_path, "r") as archive:
            members = []
            for member in archive.infolist():
                if member.is_dir():
                    continue
                basename = os.path.basename(member.filename)
                ext = os.path.splitext(basename)[1].lower()
                if not basename or ext not in allowed:
                    continue
                members.append(member)
            if not members:
                raise RuntimeError(f"No supported audio files found in zip archive: {source_path}")
            for member in sorted(members, key=lambda item: item.filename):
                digest.update(
                    f"{member.filename}|{member.file_size}|{member.CRC}|{member.date_time}".encode("utf-8")
                )
        return f"zip:{digest.hexdigest()}"

    ext = os.path.splitext(source_path)[1].lower()
    if ext in {f".{audio_ext}" for audio_ext in SUPPORTED_AUDIO}:
        return generate_stable_audio_component(audio_file_path=source_path)

    raise ValueError(f"Unsupported dataset source: {source_path}")


def _iter_audio_batches(waveform: torch.Tensor) -> Iterable[torch.Tensor]:
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)

    while waveform.dim() > 3 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)

    if waveform.dim() == 1:
        yield waveform.unsqueeze(0)
        return

    if waveform.dim() == 2:
        if waveform.shape[0] <= 8:
            yield waveform
            return
        for clip in waveform:
            yield clip.unsqueeze(0)
        return

    if waveform.dim() != 3:
        raise ValueError(f"Unsupported audio tensor shape for dataset staging: {tuple(waveform.shape)}")

    for clip in waveform:
        if clip.dim() == 1:
            yield clip.unsqueeze(0)
        else:
            yield clip


def _write_audio_clip(audio_tensor: torch.Tensor, sample_rate: int, output_path: str) -> None:
    clip = audio_tensor.detach().cpu().float()
    if clip.dim() == 1:
        clip = clip.unsqueeze(0)
    if clip.dim() != 2:
        raise ValueError(f"Expected [channels, samples] audio clip, got shape {tuple(clip.shape)}")
    clip = clip.clamp(-1.0, 1.0)
    wav_data = clip.numpy()
    if wav_data.shape[0] <= 8:
        wav_data = wav_data.T
    wavfile.write(output_path, sample_rate, wav_data.astype(np.float32))


def _flatten_audio_inputs(audio_inputs: List[Dict[str, object]], destination_dir: str, start_index: int = 0) -> int:
    os.makedirs(destination_dir, exist_ok=True)
    file_count = 0
    for input_index, audio_input in enumerate(audio_inputs):
        normalized_audio = AudioProcessingUtils.normalize_audio_input(audio_input, f"opt_audio{input_index + 1}")
        sample_rate = int(normalized_audio["sample_rate"])
        waveform = normalized_audio["waveform"]
        for clip_index, clip in enumerate(_iter_audio_batches(waveform)):
            output_name = f"{start_index + file_count:05d}_opt_audio{input_index + 1}_{clip_index + 1}.wav"
            _write_audio_clip(clip, sample_rate, os.path.join(destination_dir, output_name))
            file_count += 1
    return file_count


def _build_feature_config(device: str) -> SimpleNamespace:
    resolved_device = str(device or "cpu")
    is_half = resolved_device.startswith("cuda")
    if is_half:
        x_pad, x_query, x_center, x_max = 3, 10, 60, 64
    else:
        x_pad, x_query, x_center, x_max = 1, 6, 38, 41
    return SimpleNamespace(
        device=resolved_device,
        is_half=is_half,
        x_pad=x_pad,
        x_query=x_query,
        x_center=x_center,
        x_max=x_max,
    )


class Preprocess:
    def __init__(
        self,
        sr: int,
        exp_dir: str,
        period: float = 3.0,
        overlap: float = 0.3,
        max_volume: float = 0.95,
    ):
        self.slicer = Slicer(
            sr=sr,
            threshold=-50,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.period = period
        self.overlap = overlap
        self.tail = self.period + self.overlap
        self.max_volume = max_volume
        self.exp_dir = exp_dir
        self.gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(exp_dir, "1_16k_wavs")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def println(self, message: str) -> None:
        print(message)
        with open(os.path.join(self.exp_dir, "preprocess.log"), "a+", encoding="utf-8") as handle:
            handle.write(f"{message}\n")
            handle.flush()

    def norm_write(self, audio: np.ndarray, idx0: int, idx1: int) -> None:
        if len(audio) <= self.overlap * self.sr * 2:
            print(f"skipped short audio clip: {idx0}_{idx1}.wav ({len(audio)=})")
            return
        wavfile.write(os.path.join(self.gt_wavs_dir, f"{idx0}_{idx1}.wav"), self.sr, audio.astype(np.float32))
        remixed_audio = remix_audio((audio, self.sr), target_sr=16000, max_volume=self.max_volume)
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{idx0}_{idx1}.wav"),
            16000,
            remixed_audio[0].astype(np.float32),
        )

    def pipeline(self, path: str, idx0: int) -> None:
        try:
            input_audio = load_input_audio(path, self.sr, verbose=False)
            idx1 = 0
            for sliced in self.slicer.slice(input_audio[0]):
                i = 0
                while True:
                    start = int(self.sr * (self.period - self.overlap) * i)
                    i += 1
                    if len(sliced[start:]) > self.tail * self.sr:
                        tmp_audio = sliced[start : start + int(self.period * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = sliced[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            self.println(f"{path}->Suc.")
        except Exception:
            self.println(f"{path}->{traceback.format_exc()}")

    def pipeline_mp_inp_dir(self, inp_root: str, n_workers: int) -> None:
        infos = [
            (os.path.join(inp_root, name), idx)
            for idx, name in enumerate(sorted(os.listdir(inp_root)))
        ]
        for worker_index in range(max(n_workers, 1)):
            self.pipeline_mp(infos[worker_index:: max(n_workers, 1)])

    def pipeline_mp(self, infos: List[Tuple[str, int]]) -> None:
        for path, idx0 in infos:
            self.pipeline(path, idx0)


class FeatureInput(FeatureExtractor):
    def __init__(
        self,
        model,
        f0_method: str,
        exp_dir: str,
        config,
        samplerate: int = 16000,
        hop_size: int = 160,
        version: str = "v2",
        if_f0: bool = False,
        progress: ProgressReporter | None = None,
    ):
        self.sr = samplerate
        self.hop = hop_size
        self.f0_method = f0_method
        self.exp_dir = exp_dir
        self.device = config.device
        self.version = version
        self.if_f0 = if_f0
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = hz_to_mel(self.f0_min)
        self.f0_mel_max = hz_to_mel(self.f0_max)
        self.model = model
        self.progress = progress
        super().__init__(samplerate, config, onnx=False)

    def printt(self, message: str) -> None:
        print(message)
        with open(os.path.join(self.exp_dir, "extract_f0_feature.log"), "a+", encoding="utf-8") as handle:
            handle.write(f"{message}\n")
            handle.flush()

    def compute_feats(self, audio: np.ndarray):
        feats = torch.from_numpy(audio).float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.half().to(self.device) if self.device not in ["mps", "cpu"] else feats.to(self.device),
            "padding_mask": padding_mask.to(self.device),
            "output_layer": 9 if self.version == "v1" else 12,
        }
        feats = self.model.extract_features(version=self.version, **inputs)
        feats = feats.squeeze(0).float().cpu().numpy()
        if np.isnan(feats).sum() == 0:
            return feats
        self.printt("==contains nan==")
        return None

    def compute_f0(self, audio: np.ndarray):
        return self.get_f0(audio, 0, self.f0_method, crepe_hop_length=self.hop, verbose=False)

    def go(self, paths: List[Tuple[str, str, str, str]]) -> None:
        if not paths:
            return
        for idx, (inp_path, opt_path1, opt_path2, opt_path3) in enumerate(paths):
            try:
                if (
                    os.path.exists(opt_path1 + ".npy")
                    and os.path.exists(opt_path2 + ".npy")
                    and os.path.exists(opt_path3 + ".npy")
                ):
                    if self.progress is not None:
                        self.progress.advance(skipped=True)
                    continue
                audio, _ = load_input_audio(inp_path, self.sr, verbose=False)
                feats = self.compute_feats(audio)
                if feats is not None:
                    np.save(opt_path3, feats, allow_pickle=False)
                    if self.if_f0:
                        coarse_pit, feature_pit = self.compute_f0(audio)
                        np.save(opt_path2, feature_pit, allow_pickle=False)
                        np.save(opt_path1, coarse_pit, allow_pickle=False)
                if self.progress is not None:
                    self.progress.advance()
            except Exception:
                self.printt(f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}")


def preprocess_trainset(
    inp_root: str,
    sr: int,
    n_workers: int,
    exp_dir: str,
    period: float = 3.0,
    overlap: float = 0.3,
    max_volume: float = 1.0,
) -> None:
    processor = Preprocess(sr, exp_dir, period=period, overlap=overlap, max_volume=max_volume)
    processor.println("start preprocess")
    processor.pipeline_mp_inp_dir(inp_root, n_workers)
    processor.println("end preprocess")
    del processor
    gc_collect()


def extract_features_trainset(
    hubert_model,
    exp_dir: str,
    n_workers: int,
    f0_method: str,
    device: str,
    version: str,
    if_f0: bool,
    crepe_hop_length: int,
) -> None:
    inp_root = os.path.join(exp_dir, "1_16k_wavs")
    total_clips = len(
        [
            name
            for name in os.listdir(inp_root)
            if "spec" not in os.path.join(inp_root, name)
        ]
    )
    progress = ProgressReporter("RVC feature extraction", total_clips) if total_clips else None
    feature_input = FeatureInput(
        model=hubert_model,
        f0_method=f0_method,
        exp_dir=exp_dir,
        config=_build_feature_config(device),
        version=version,
        if_f0=if_f0,
        hop_size=crepe_hop_length,
        progress=progress,
    )
    opt_root1 = os.path.join(exp_dir, "2a_f0")
    opt_root2 = os.path.join(exp_dir, "2b-f0nsf")
    opt_root3 = os.path.join(exp_dir, "3_feature256" if version == "v1" else "3_feature768")
    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    os.makedirs(opt_root3, exist_ok=True)

    paths = []
    for name in sorted(os.listdir(inp_root)):
        inp_path = os.path.join(inp_root, name)
        if "spec" in inp_path:
            continue
        paths.append(
            [
                inp_path,
                os.path.join(opt_root1, ",".join([str(f0_method), name])),
                os.path.join(opt_root2, ",".join([str(f0_method), name])),
                os.path.join(opt_root3, ",".join([str(f0_method), name])),
            ]
        )

    threads = []
    n_workers = max(n_workers, 1)
    device_name = str(device)
    if device_name.startswith("cuda"):
        for idx in range(n_workers):
            feature_input.go(paths[idx::n_workers])
    else:
        for idx in range(n_workers):
            thread = Thread(target=feature_input.go, args=(paths[idx::n_workers],), daemon=True)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()


def build_training_filelist(dataset_dir: str, sample_rate: str, f0_method: str, mute_ratio: float) -> str:
    feature_dir = os.path.join(dataset_dir, "3_feature768")
    gt_wavs_dir = os.path.join(dataset_dir, "0_gt_wavs")
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"Missing feature directory: {feature_dir}")

    use_f0 = bool(f0_method)
    if use_f0:
        f0_dir = os.path.join(dataset_dir, "2a_f0")
        f0nsf_dir = os.path.join(dataset_dir, "2b-f0nsf")
        names = (
            {os.path.splitext(name)[0] for name in os.listdir(feature_dir)}
            & {os.path.splitext(name)[0] for name in os.listdir(f0_dir)}
            & {os.path.splitext(name)[0] for name in os.listdir(f0nsf_dir)}
        )
    else:
        names = {os.path.splitext(name)[0] for name in os.listdir(feature_dir)}

    entries = []
    missing_ground_truth = []
    for name in sorted(names):
        name_parts = name.split(",")
        gt_name = name if len(name_parts) == 1 else name_parts[-1]
        gt_file = os.path.join(gt_wavs_dir, gt_name)
        if not os.path.isfile(gt_file):
            missing_ground_truth.append(gt_name)
            continue

        if use_f0:
            entry = "|".join(
                [
                    gt_file,
                    os.path.join(feature_dir, f"{name}.npy"),
                    os.path.join(f0_dir, f"{name}.npy"),
                    os.path.join(f0nsf_dir, f"{name}.npy"),
                    "0",
                ]
            )
        else:
            entry = "|".join([gt_file, os.path.join(feature_dir, f"{name}.npy"), "0"])
        entries.append(entry)

    if missing_ground_truth:
        raise RuntimeError(f"Missing ground truth wav files: {missing_ground_truth[:5]}")

    # Keep at least two mute samples to match both the Comfy-RVC reference node
    # and the upstream RVC WebUI training filelist behavior.
    mute_count = max(2, int(len(entries) * mute_ratio))
    mute_feature_dim = 768
    for _ in range(mute_count):
        if use_f0:
            entry = "|".join(
                [
                    str(MUTE_DATASET_DIR / "0_gt_wavs" / f"mute{sample_rate}.wav"),
                    str(MUTE_DATASET_DIR / f"3_feature{mute_feature_dim}" / "mute.npy"),
                    str(MUTE_DATASET_DIR / "2a_f0" / "mute.wav.npy"),
                    str(MUTE_DATASET_DIR / "2b-f0nsf" / "mute.wav.npy"),
                    "0",
                ]
            )
        else:
            entry = "|".join(
                [
                    str(MUTE_DATASET_DIR / "0_gt_wavs" / f"mute{sample_rate}.wav"),
                    str(MUTE_DATASET_DIR / f"3_feature{mute_feature_dim}" / "mute.npy"),
                    "0",
                ]
            )
        entries.append(entry)

    np.random.shuffle(entries)
    filelist_path = os.path.join(dataset_dir, "filelist.txt")
    with open(filelist_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(entries))
    return filelist_path


def prepare_rvc_training_dataset(
    shared_settings: Dict[str, str],
    dataset_source: str,
    model_name: str,
    sample_rate: str = "40k",
    cpu_workers: int = 1,
    chunk_seconds: float = 3.0,
    overlap_seconds: float = 0.3,
    max_volume: float = 0.95,
    mute_ratio: float = 0.0,
    reuse_existing: bool = True,
    audio_inputs: List[Dict[str, object]] | None = None,
) -> Dict[str, str]:
    audio_inputs = audio_inputs or []
    source_path = None
    if dataset_source and str(dataset_source).strip():
        source_path = _resolve_source_path(dataset_source)
    if source_path is None and not audio_inputs:
        raise ValueError("Provide dataset_source or connect at least one opt_audio input")

    safe_model_name = _slugify(model_name)
    f0_method = shared_settings.get("f0_method", "rmvpe")
    crepe_hop_length = int(shared_settings.get("crepe_hop_length", 160))
    device = shared_settings.get("device", "cpu")
    hubert_path = shared_settings.get("hubert_path")
    hubert_model_name = shared_settings.get("hubert_model", "content-vec-best")

    if not hubert_path or not os.path.exists(hubert_path):
        hubert_path = ensure_hubert_model(hubert_model_name)
    if not hubert_path or not os.path.exists(hubert_path):
        raise FileNotFoundError("HuBERT model could not be resolved for RVC dataset preparation")

    hash_parts = [
        sample_rate,
        f0_method,
        str(crepe_hop_length),
        str(chunk_seconds),
        str(overlap_seconds),
        str(max_volume),
        str(mute_ratio),
    ]
    if source_path is not None:
        hash_parts.append(_fingerprint_source_path(source_path))
    for audio_input in audio_inputs:
        hash_parts.append(generate_stable_audio_component(reference_audio=audio_input))
    dataset_hash = hashlib.md5("|".join(hash_parts).encode()).hexdigest()[:12]
    output_root = os.path.join(
        folder_paths.get_output_directory(),
        "tts_audio_suite_training",
        "rvc",
        "datasets",
        f"dataset_{dataset_hash}",
    )
    raw_input_dir = os.path.join(output_root, "source_audio")
    filelist_path = os.path.join(output_root, "filelist.txt")

    if not os.path.isfile(filelist_path) or not reuse_existing:
        if not reuse_existing and os.path.isdir(output_root):
            shutil.rmtree(output_root)
        os.makedirs(output_root, exist_ok=True)
        file_count = 0
        if source_path is not None:
            file_count += _flatten_audio_source(source_path, raw_input_dir)
        if audio_inputs:
            file_count += _flatten_audio_inputs(audio_inputs, raw_input_dir, start_index=file_count)
        preprocess_trainset(
            raw_input_dir,
            SR_MAP[sample_rate],
            cpu_workers,
            output_root,
            period=chunk_seconds,
            overlap=overlap_seconds,
            max_volume=max_volume,
        )

        hubert_model = load_hubert(hubert_path, SimpleNamespace(device=device))
        try:
            extract_features_trainset(
                hubert_model,
                output_root,
                cpu_workers,
                f0_method=f0_method,
                device=device,
                version="v2",
                if_f0=bool(f0_method),
                crepe_hop_length=crepe_hop_length,
            )
        finally:
            del hubert_model
            gc_collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        filelist_path = build_training_filelist(output_root, sample_rate, f0_method, mute_ratio)
    else:
        file_count = len(list(_iter_audio_files(raw_input_dir))) if os.path.isdir(raw_input_dir) else 0

    source_types = []
    if source_path is not None:
        source_types.append("path")
    if audio_inputs:
        source_types.append(f"audio_inputs:{len(audio_inputs)}")

    return {
        "type": "training_dataset",
        "engine_type": "rvc",
        "training_mode": "voice_model",
        "model_name": safe_model_name,
        "dataset_source": source_path or "",
        "dataset_dir": output_root,
        "training_files": filelist_path,
        "sample_rate": sample_rate,
        "if_f0": bool(f0_method),
        "f0_method": f0_method,
        "hubert_path": hubert_path,
        "hubert_model": hubert_model_name,
        "crepe_hop_length": crepe_hop_length,
        "device": str(device),
        "file_count": file_count,
        "audio_input_count": len(audio_inputs),
        "source_summary": "+".join(source_types) if source_types else "unknown",
    }
