from collections import OrderedDict
import glob
import json
import os
import shutil
import traceback
from typing import Tuple
import warnings
import numpy as np

from tqdm import tqdm
from .lib.audio import SR_MAP
from .lib.train import utils
import datetime
from random import shuffle, randint
import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from .lib.infer_pack import commons
from time import sleep
from time import time as ttime
from .lib.train.data_utils import (
    BucketSampler,
    TextAudioLoaderMultiNSFsid,
    TextAudioLoader,
    TextAudioCollateMultiNSFsid,
    TextAudioCollate,
    DistributedBucketSampler,
)
from .lib.train.losses import LossBalancer, MultiScaleMelSpectrogramLoss, combined_aux_loss, generator_loss, discriminator_loss, feature_loss, gradient_norm_loss, kl_loss
from .lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from engines.training.progress_io import write_json_progress_file

try:
    import comfy.model_management as comfy_model_management
except Exception:
    comfy_model_management = None


warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.GradScaler\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)

INTERRUPT_GRACE_PERIOD_SEC = 30.0


def _checkpoint_sort_key(path):
    basename = os.path.basename(path)
    stem, _ = os.path.splitext(basename)
    if stem.endswith("_latest"):
        return (2, float("inf"), os.path.getmtime(path))
    digits = "".join(filter(str.isdigit, stem))
    if digits:
        return (1, int(digits), os.path.getmtime(path))
    return (0, 0, os.path.getmtime(path))


def _prune_checkpoint_history(model_dir, max_checkpoints, logger_obj=None):
    if int(max_checkpoints) <= 0:
        return

    for prefix in ("G", "D"):
        candidates = glob.glob(os.path.join(model_dir, f"{prefix}_*.pth"))
        candidates.sort(key=_checkpoint_sort_key)
        stale_candidates = candidates[:-int(max_checkpoints)]
        for stale_path in stale_candidates:
            try:
                os.remove(stale_path)
                if logger_obj is not None:
                    logger_obj.info("Pruned old checkpoint | file=%s", os.path.basename(stale_path))
            except FileNotFoundError:
                continue
            except Exception as error:
                if logger_obj is not None:
                    logger_obj.warning(
                        "Failed to prune old checkpoint %s: %s",
                        os.path.basename(stale_path),
                        error,
                    )

def save_checkpoint(ckpt, name, epoch, hps, model_path=None):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = "%sepoch" % epoch
        opt["sr"] = hps.sample_rate
        opt["f0"] = hps.if_f0
        opt["version"] = hps.version
        if model_path is None: model_path=os.path.join(hps.model_dir,name+".pth")
        torch.save(opt, model_path)
        return "Success."
    except:
        return traceback.format_exc()

class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def snapshot(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})", elapsed_time

    def record(self):
        return self.snapshot()[0]


def _count_training_samples(training_filelist):
    try:
        with open(training_filelist, "r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return 0


def _log_training_banner(logger, hps, steps_per_epoch):
    sample_count = _count_training_samples(hps.data.training_files)
    clip_info = f"{sample_count} clips" if sample_count > 0 else "unknown clip count"
    device_label = f"cuda:{hps.gpus}" if hps.gpus else ("mps" if torch.backends.mps.is_available() else "cpu")
    pretrain_g = os.path.basename(str(getattr(hps, "pretrainG", "") or "")) or "none"
    pretrain_d = os.path.basename(str(getattr(hps, "pretrainD", "") or "")) or "none"
    continue_from_model = os.path.basename(str(getattr(hps, "continue_from_model_path", "") or "")) or ""
    logger.info(
        "RVC training started | model=%s | sr=%s | %s | epochs=%s | steps/epoch=%s | "
        "batch=%s | lr=%.2E | f0=%s | workers=%s | device=%s",
        hps.name,
        hps.sample_rate,
        clip_info,
        hps.total_epoch,
        steps_per_epoch,
        hps.train.batch_size,
        hps.train.learning_rate,
        "on" if hps.if_f0 else "off",
        hps.train.num_workers,
        device_label,
    )
    logger.info(
        "Init | pretrainG=%s | pretrainD=%s",
        pretrain_g,
        pretrain_d,
    )
    if continue_from_model:
        logger.info("Continue from model | generator=%s", continue_from_model)
    logger.info(
        "Outputs | job_dir=%s | final_model=%s",
        hps.model_dir,
        hps.model_path,
    )


def _extract_initial_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model"), dict):
            return checkpoint["model"], True
        if isinstance(checkpoint.get("weight"), dict):
            return checkpoint["weight"], False
    if isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, False
    raise ValueError("Unsupported checkpoint format for initialization")


def _load_initial_checkpoint(model, checkpoint_path, logger=None, label="model"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, strict = _extract_initial_state_dict(checkpoint)
    target_model = model.module if hasattr(model, "module") else model
    result = target_model.load_state_dict(state_dict, strict=strict)
    if logger is not None:
        mode = "strict" if strict else "warm-start"
        logger.info("Loaded %s init checkpoint (%s): %s", label, mode, checkpoint_path)
        logger.info(result)
    else:
        print(f"Loaded {label} init checkpoint ({'strict' if strict else 'warm-start'}): {checkpoint_path}")
        print(result)
    return result


def _serialize_multiscale_mel_state(hps):
    if not hps.train.get("use_multiscale"):
        return None

    multiscale = globals().get("MultiscaleMelLoss")
    if multiscale is None or not hasattr(multiscale, "to_dict"):
        return None
    return multiscale.to_dict()


def _log_epoch_summary(logger, epoch, hps, global_step, lr, epoch_time, metrics):
    progress = (100.0 * epoch / max(hps.total_epoch, 1))
    parts = [
        f"Epoch {epoch}/{hps.total_epoch}",
        f"{progress:.0f}%",
        f"step {global_step}",
        f"lr {lr:.2E}",
        f"time {epoch_time}",
        f"loss {metrics['total_loss']:.3f}",
        f"gen {metrics['loss_gen_all']:.3f}",
        f"disc {metrics['loss_disc_all']:.3f}",
        f"mel {metrics['loss_mel']:.3f}",
        f"kl {metrics['loss_kl']:.3f}",
        f"fm {metrics['loss_fm']:.3f}",
    ]
    if metrics["aux_loss"] > 0:
        parts.append(f"aux {metrics['aux_loss']:.3f}")
    logger.info(" | ".join(parts))


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    try:
        return float(value)
    except Exception:
        return 0.0


class TrainingProgressTracker:
    def __init__(self, hps):
        self.path = getattr(hps, "progress_file", "") or ""
        self.enabled = bool(self.path)
        self.history = []
        self.recent_loss_trace = []
        self.last_write = 0.0
        self.epoch_started_at = None
        self.training_started_at = None
        self.best_gen_loss = None
        self.sample_count = _count_training_samples(hps.data.training_files)
        self.hps = hps
        self.last_write_warning = 0.0

    def _write(self, payload):
        if not self.enabled:
            return
        if write_json_progress_file(self.path, payload):
            self.last_write = ttime()
            return

        now = ttime()
        if (now - self.last_write_warning) >= 5.0:
            print("RVC training: progress.json update skipped because the file is temporarily locked.")
            self.last_write_warning = now

    def initialize(self, steps_per_epoch):
        total_steps = max(int(self.hps.total_epoch) * max(int(steps_per_epoch), 1), 1)
        self.training_started_at = ttime()
        self._write(
            {
                "status": "running",
                "phase": "training",
                "node_id": str(getattr(self.hps, "node_id", "") or ""),
                "engine_type": "rvc",
                "model_name": self.hps.name,
                "sample_rate": self.hps.sample_rate,
                "total_epochs": int(self.hps.total_epoch),
                "steps_per_epoch": int(steps_per_epoch),
                "total_steps": int(total_steps),
                "completed_total_steps": 0,
                "batch_size": int(self.hps.train.batch_size),
                "learning_rate": float(self.hps.train.learning_rate),
                "clip_count": int(self.sample_count),
                "workers": int(self.hps.train.num_workers),
                "started_at": float(self.training_started_at),
                "current_epoch": 0,
                "epoch_progress": 0.0,
                "overall_progress": 0.0,
                "history": [],
                "recent_loss_trace": [],
                "updated_at": datetime.datetime.now().isoformat(),
            }
        )

    def start_epoch(self, epoch, steps_per_epoch, global_step, lr):
        if self.training_started_at is None:
            self.training_started_at = ttime()
        self.epoch_started_at = ttime()
        completed_total_steps = max((int(epoch) - 1) * max(int(steps_per_epoch), 1), 0)
        self._write(
            {
                **self._read_current(),
                "status": "running",
                "phase": "training",
                "current_epoch": int(epoch),
                "steps_per_epoch": int(steps_per_epoch),
                "total_steps": int(max(int(self.hps.total_epoch) * max(int(steps_per_epoch), 1), 1)),
                "completed_total_steps": int(completed_total_steps),
                "current_step": 0,
                "global_step": int(global_step),
                "epoch_progress": 0.0,
                "overall_progress": float((epoch - 1) / max(self.hps.total_epoch, 1)),
                "learning_rate": float(lr),
                "started_at": float(self.training_started_at),
                "updated_at": datetime.datetime.now().isoformat(),
            }
        )

    def _read_current(self):
        if not self.enabled or not os.path.isfile(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def update_batch(self, epoch, processed_step_idx, steps_per_epoch, global_step, lr, metrics, source_batch_idx=None):
        if not self.enabled:
            return

        now = ttime()
        completed_steps = int(processed_step_idx) + 1
        total_steps = max(int(self.hps.total_epoch) * max(int(steps_per_epoch), 1), 1)
        completed_total_steps = ((int(epoch) - 1) * max(int(steps_per_epoch), 1)) + completed_steps
        elapsed = max(now - (self.epoch_started_at or now), 1e-6)
        epoch_avg_step_time = elapsed / max(completed_steps, 1)
        total_elapsed = max(now - (self.training_started_at or now), 1e-6)
        overall_avg_step_time = total_elapsed / max(completed_total_steps, 1)
        blended_step_time = (overall_avg_step_time * 0.7) + (epoch_avg_step_time * 0.3)
        force_write = completed_steps in {1, steps_per_epoch}
        if not force_write and (now - self.last_write) < 0.3:
            return

        total_loss = _as_float(metrics.get("loss_gen_all", 0.0)) + _as_float(metrics.get("loss_disc_all", 0.0))
        trace_entry = {
            "epoch": int(epoch),
            "step": int(completed_steps),
            "global_step": int(global_step),
            "total_loss": float(total_loss),
        }
        if source_batch_idx is not None:
            trace_entry["source_batch_idx"] = int(source_batch_idx)
        self.recent_loss_trace.append(trace_entry)
        self.recent_loss_trace = self.recent_loss_trace[-120:]

        overall_progress = completed_total_steps / max(total_steps, 1)
        remaining_steps = max(total_steps - completed_total_steps, 0)
        payload = {
            **self._read_current(),
            "status": "running",
            "phase": "training",
            "current_epoch": int(epoch),
            "current_step": int(completed_steps),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
            "completed_total_steps": int(completed_total_steps),
            "global_step": int(global_step),
            "epoch_progress": float(completed_steps / max(steps_per_epoch, 1)),
            "overall_progress": float(overall_progress),
            "it_per_sec": float(1.0 / blended_step_time) if blended_step_time > 0 else 0.0,
            "step_time_sec": float(blended_step_time),
            "eta_sec": float(blended_step_time * remaining_steps),
            "learning_rate": float(lr),
            "current_metrics": {key: _as_float(value) for key, value in metrics.items()},
            "recent_loss_trace": self.recent_loss_trace,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        self._write(payload)

    def end_epoch(self, epoch, global_step, lr, epoch_time_sec, metrics):
        epoch_entry = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "learning_rate": float(lr),
            "epoch_time_sec": float(epoch_time_sec),
            **{key: _as_float(value) for key, value in metrics.items()},
        }
        self.history.append(epoch_entry)
        self.history = self.history[-60:]

        gen_loss = epoch_entry.get("loss_gen_all", 0.0)
        if self.best_gen_loss is None or gen_loss < self.best_gen_loss:
            self.best_gen_loss = gen_loss

        payload = {
            **self._read_current(),
            "status": "running",
            "phase": "training",
            "current_epoch": int(epoch),
            "current_step": int(epoch_entry.get("steps_per_epoch", 0)),
            "completed_total_steps": int(epoch * max(int(epoch_entry.get("steps_per_epoch", 0)), 1)),
            "global_step": int(global_step),
            "epoch_progress": 1.0,
            "overall_progress": float(epoch / max(self.hps.total_epoch, 1)),
            "learning_rate": float(lr),
            "best_gen_loss": float(self.best_gen_loss),
            "last_epoch": epoch_entry,
            "history": self.history,
            "recent_loss_trace": self.recent_loss_trace,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        self._write(payload)

    def finalize(self, status, phase, **extra):
        payload = {
            **self._read_current(),
            "status": status,
            "phase": phase,
            "history": self.history,
            "recent_loss_trace": self.recent_loss_trace,
            "best_gen_loss": self.best_gen_loss,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        payload.update(extra)
        self._write(payload)


def _cancel_flag_path(hps):
    return str(getattr(hps, "cancel_flag_path", "") or "").strip()


def _clear_cancel_flag(hps):
    path = _cancel_flag_path(hps)
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _write_cancel_flag(hps, reason="cancelled"):
    path = _cancel_flag_path(hps)
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "reason": reason,
                "updated_at": datetime.datetime.now().isoformat(),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


def _cancel_requested(hps):
    path = _cancel_flag_path(hps)
    return bool(path) and os.path.isfile(path)


def _parent_interrupt_requested():
    if comfy_model_management is None:
        return False
    try:
        return bool(comfy_model_management.processing_interrupted())
    except Exception:
        return bool(getattr(comfy_model_management, "interrupt_processing", False))


def _raise_if_cancel_requested(hps, progress_tracker=None):
    if not _cancel_requested(hps):
        return
    if progress_tracker is not None:
        progress_tracker.finalize(
            "cancelled",
            "cancelled",
            message="Training cancelled by user",
        )
    raise InterruptedError("RVC training interrupted by user")


def _terminate_children(children):
    for child in children.values():
        if child.is_alive():
            child.terminate()

    deadline = ttime() + 5.0
    while ttime() < deadline:
        any_alive = False
        for child in children.values():
            child.join(timeout=0.1)
            if child.is_alive():
                any_alive = True
        if not any_alive:
            break

    for child in children.values():
        if child.is_alive() and hasattr(child, "kill"):
            child.kill()
            child.join(timeout=0.1)


def _wait_for_children(children, timeout_sec):
    deadline = ttime() + max(float(timeout_sec), 0.0)
    while ttime() < deadline:
        any_alive = False
        for child in children.values():
            child.join(timeout=0.1)
            if child.is_alive():
                any_alive = True
        if not any_alive:
            return True
    return not any(child.is_alive() for child in children.values())

def train_model(hps: "utils.HParams"):
    os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(8189, 8205+hps.train.num_workers**2))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

    n_gpus = len(hps.gpus.split("-")) if hps.gpus else torch.cuda.device_count()

    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        n_gpus = 1
    if n_gpus < 1:
        # patch to unblock people without gpus. there is probably a better way.
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1

    _clear_cancel_flag(hps)
    gpu_devices = hps.gpus.split("-") if hps.gpus else range(n_gpus)

    children = {}
    spawn_ctx = mp.get_context("spawn")
    try:
        for i, device in enumerate(gpu_devices):
            subproc = spawn_ctx.Process(
                target=run,
                args=(
                    i,
                    n_gpus,
                    hps,
                    device
                ),
            )
            children[i] = subproc
            subproc.start()

        while True:
            any_alive = False
            for child in children.values():
                child.join(timeout=0.25)
                if child.is_alive():
                    any_alive = True

            if _parent_interrupt_requested():
                _write_cancel_flag(hps, reason="comfyui_interrupt")
                if not _wait_for_children(children, INTERRUPT_GRACE_PERIOD_SEC):
                    _terminate_children(children)
                raise InterruptedError("RVC training interrupted by user")

            if not any_alive:
                break

        if _cancel_requested(hps):
            raise InterruptedError("RVC training interrupted by user")

        bad_exit_codes = [
            child.exitcode for child in children.values()
            if child.exitcode not in (0, None)
        ]
        if bad_exit_codes:
            raise RuntimeError(f"RVC training worker failed with exit codes: {bad_exit_codes}")
    finally:
        if any(child.is_alive() for child in children.values()):
            _terminate_children(children)

def run(rank, n_gpus, hps, device):
    global global_step, least_loss, loss_file, best_model_name, MultiscaleMelLoss
    global_step = 0
    loss_file = os.path.join(hps.model_dir,"losses.json")
    progress_tracker = TrainingProgressTracker(hps) if rank == 0 else None

    if os.path.isfile(loss_file):
        with open(loss_file,"r") as f:
            data: dict = json.load(f)
            least_loss = data.get("least_loss",hps.best_model_threshold)
            best_model_name = data.get("best_model_name","")
    else:
        least_loss = hps.best_model_threshold
        best_model_name = ""

    def _save_interrupt_resume_checkpoint(epoch, net_g, net_d, optim_g, optim_d, balancer_g, balancer_d, logger_obj=None):
        g_name = f"G_{int(epoch)}.pth"
        d_name = f"D_{int(epoch)}.pth"
        g_path = os.path.join(hps.model_dir, g_name)
        d_path = os.path.join(hps.model_dir, d_name)
        lr = float(optim_g.param_groups[0]["lr"])
        msml_state = _serialize_multiscale_mel_state(hps)
        try:
            utils.save_checkpoint(
                net_g,
                optim_g,
                lr,
                int(epoch),
                g_path,
                balancer=balancer_g.to_dict(),
                msml=msml_state,
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                lr,
                int(epoch),
                d_path,
                balancer=balancer_d.to_dict(),
            )
            message = (
                f"Saved interrupt resume checkpoints | "
                f"G={os.path.basename(g_path)} | D={os.path.basename(d_path)}"
            )
            _prune_checkpoint_history(hps.model_dir, getattr(hps, "max_checkpoints", 1), logger_obj=logger_obj)
            if logger_obj is not None:
                logger_obj.info(message)
            else:
                print(message)
            return g_path, d_path
        except Exception as error:
            message = f"Failed to save interrupt resume checkpoint: {error}"
            if logger_obj is not None:
                logger_obj.warning(message)
            else:
                print(message)
            return "", ""

    if hps.version == "v1":
        from .lib.infer_pack.models import (
            SynthesizerTrnMs256NSFsid as RVC_Model_f0,
            SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
            MultiPeriodDiscriminator,
        )
    else:
        from .lib.infer_pack.models import (
            SynthesizerTrnMs768NSFsid as RVC_Model_f0,
            SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
            MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
        )

    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        # writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    if n_gpus>1:
        try:
            dist.init_process_group(backend="gloo", init_method="env://", world_size=n_gpus, rank=rank)
            distributed = True
        except Exception as error:
            print(f"Failed to initialize dist: {error=}")
            distributed = False
    else: distributed=False
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{device}")

    if hps.if_f0:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    
    if distributed:
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size * n_gpus,
            [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
    else:
        train_sampler = BucketSampler(
            train_dataset,
            hps.train.batch_size,
            [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s            
            shuffle=True,
        )
    
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=hps.train.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        **(
            {
                "persistent_workers": True,
                "prefetch_factor": 8,
            }
            if hps.train.num_workers > 0
            else {}
        ),
    )
    hps.sync_log_interval(len(train_loader))
    if rank == 0:
        _log_training_banner(logger, hps, len(train_loader))
        progress_tracker.initialize(len(train_loader))
    _raise_if_cancel_requested(hps, progress_tracker if rank == 0 else None)
    
    if hps.if_f0:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if distributed:
        if torch.cuda.is_available():
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])
        else:
            net_g = DDP(net_g)
            net_d = DDP(net_d)

    try:  # resume training
        d_checkpoint = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        g_checkpoint = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        if not d_checkpoint or not g_checkpoint:
            raise FileNotFoundError("no existing checkpoints")

        _, _, _, epoch_str, d_kwargs = utils.load_checkpoint(d_checkpoint, net_d, optim_d)
        if rank == 0:
            logger.info("Resumed discriminator from %s", os.path.basename(d_checkpoint))
        
        _, _, _, epoch_str, g_kwargs = utils.load_checkpoint(g_checkpoint, net_g, optim_g)
        if rank == 0:
            logger.info("Resumed generator from %s", os.path.basename(g_checkpoint))

        global_step = (epoch_str - 1) * len(train_loader)

    except FileNotFoundError:
        if rank == 0:
            logger.info("Starting fresh training run")
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            _load_initial_checkpoint(net_g, hps.pretrainG, logger if rank == 0 else None, "generator")
        if hps.pretrainD != "":
            _load_initial_checkpoint(net_d, hps.pretrainD, logger if rank == 0 else None, "discriminator")
        d_kwargs = g_kwargs = {}
    except Exception as e:
        logger.warning(f"Checkpoint resume failed, starting fresh: {e}")
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            _load_initial_checkpoint(net_g, hps.pretrainG, logger if rank == 0 else None, "generator")
        if hps.pretrainD != "":
            _load_initial_checkpoint(net_d, hps.pretrainD, logger if rank == 0 else None, "discriminator")
        d_kwargs = g_kwargs = {}

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    if hps.train.get("use_multiscale"):
        try:
            msml_dict = g_kwargs["msml"]
            MultiscaleMelLoss = MultiScaleMelSpectrogramLoss(**msml_dict)
        except Exception as e:
            logger.error(f"Failed to load MultiScaleMelSpectrogramLoss state: {e}")
            MultiscaleMelLoss = MultiScaleMelSpectrogramLoss(
                hps.data.sampling_rate,
                adjustment_factor=min(1./len(train_loader),.05),
                epsilon=hps.train.eps)

    try:
        balancer_state = g_kwargs["balancer"]
        if rank == 0 and hps.train.get("use_balancer",False):
            logger.info(f"Using existing generator balancer state")
        balancer_g = LossBalancer(net_g,**balancer_state)
    except Exception as e:
        if rank == 0 and hps.train.get("use_balancer",False):
            logger.info(f"Starting generator balancer with fresh state")
        balancer_g = LossBalancer(
            net_g,            
            weights_decay=.5 / (1 + np.exp(-10 * (epoch_str / hps.total_epoch - 0.16)))+.5, #sigmoid scaled ema .8 at 20% epoch
            loss_decay=.8,
            epsilon=hps.train.eps,
            active=hps.train.get("use_balancer",False),
            use_pareto=hps.train.get("use_pareto",False),
            use_norm=not hps.train.get("fast_mode",False),
            initial_weights=dict(
                loss_gen=hps.train.get("c_adv",1.),
                loss_fm=hps.train.get("c_fm",2.),
                loss_mel=hps.train.get("c_mel",45.),
                loss_kl=hps.train.get("c_kl",1.),
                harmonic_loss=hps.train.get("c_hd",0.),
                tsi_loss=hps.train.get("c_tsi",0.),
                tefs_loss=hps.train.get("c_tefs",0.),
            ))
        
    try:
        balancer_state = d_kwargs["balancer"]
        if rank == 0 and hps.train.get("use_balancer",False):
            logger.info(f"Using existing discriminator balancer state")
        balancer_d = LossBalancer(net_d,**balancer_state)
    except Exception as e:
        if rank == 0 and hps.train.get("use_balancer",False):
            logger.info(f"Starting discriminator balancer with fresh state")
        balancer_d = LossBalancer(
            net_d,            
            weights_decay=commons.sigmoid_value(global_step,total_steps=10000,start_value=.5, end_value=.999, midpoint=.2),
            loss_decay=.8,
            epsilon=hps.train.eps,
            active=hps.train.get("use_balancer",False),
            use_pareto=hps.train.get("use_pareto",False),
            use_norm=not hps.train.get("fast_mode",False),
            initial_weights=dict(
                loss_disc=hps.train.get("c_adv",1.),
                gradient_penalty=hps.train.get("c_gp",0.),
            ))
    cache = []
    current_epoch = max(int(epoch_str), 1)
    try:
        for epoch in range(epoch_str, hps.train.epochs + 1):
            current_epoch = int(epoch)
            _raise_if_cancel_requested(hps, progress_tracker if rank == 0 else None)
            train_loader.batch_sampler.set_epoch(epoch)
            if rank == 0:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d],
                    [optim_g, optim_d],
                    [scheduler_g, scheduler_d],
                    scaler,
                    [train_loader, None],
                    logger,
                    [writer, None],
                    cache,
                    [balancer_g, balancer_d],
                    progress_tracker,
                )
            else:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d],
                    [optim_g, optim_d],
                    [scheduler_g, scheduler_d],
                    scaler,
                    [train_loader, None],
                    None,
                    None,
                    cache,
                    [balancer_g, balancer_d],
                    None,
                )
            scheduler_g.step()
            scheduler_d.step()
    except InterruptedError:
        interrupt_g_path = ""
        interrupt_d_path = ""
        if rank == 0:
            interrupt_g_path, interrupt_d_path = _save_interrupt_resume_checkpoint(
                current_epoch,
                net_g,
                net_d,
                optim_g,
                optim_d,
                balancer_g,
                balancer_d,
                logger,
            )
            if progress_tracker is not None:
                progress_tracker.finalize(
                    "cancelled",
                    "cancelled",
                    message="Training cancelled by user",
                    interrupt_generator_checkpoint=interrupt_g_path,
                    interrupt_discriminator_checkpoint=interrupt_d_path,
                )
        return


def train_and_evaluate(
    rank, epoch, hps, nets, optims, _, scaler, loaders, logger, writers, cache, balancer: Tuple["LossBalancer","LossBalancer"], progress_tracker=None
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, _ = loaders
    if writers is not None:
        writer, _ = writers

    global global_step, least_loss, loss_file, best_model_name, gradient_clip_value, MultiscaleMelLoss

    net_g.train()
    net_d.train()
    balancer_g, balancer_d = balancer
    gradient_clip_value = commons.sigmoid_value(global_step, total_steps=10000, start_value=1, end_value=500, midpoint=.2)
    lr = optims[0].param_groups[0]["lr"]

    # Prepare data iterator
    if hps.if_cache_data_in_gpu:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                _raise_if_cancel_requested(hps, progress_tracker if rank == 0 else None)
                # Unpack
                if hps.if_f0:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                # Load on CUDA
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                # Cache on list
                if hps.if_f0:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    steps_per_epoch = len(train_loader)
    if progress_tracker is not None:
        progress_tracker.start_epoch(epoch, steps_per_epoch, global_step, lr)

    for processed_step_idx, batch in enumerate(
        tqdm(data_iterator, desc=f"Epoch {epoch}/{hps.total_epoch}", leave=False),
        start=0,
    ):
        _raise_if_cancel_requested(hps, progress_tracker if rank == 0 else None)
        batch_idx, info = batch
        # Data
        ## Unpack
        if hps.if_f0:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if (not hps.if_cache_data_in_gpu) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            # if hps.train.fp16_run: y_hat_mel = y_hat_mel.half()
            wave_orig = wave.clone()
            wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            gen_wave = y_hat.detach()
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, gen_wave)
            
            with autocast(enabled=False):
                gradient_penalty = gradient_norm_loss(wave,gen_wave, net_d, eps=hps.train.eps)*hps.train.c_gp if hps.train.get("c_gp",0.)>0 else 0
                loss_disc, losses_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = balancer_d.on_train_batch_start(dict(
                    loss_disc=loss_disc,
                    gradient_penalty=gradient_penalty                    
                    ),input=y_hat)

            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), gradient_clip_value, batch_size=hps.train.batch_size)
            scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                if hps.train.get("use_multiscale"): loss_mel = MultiscaleMelLoss(y_hat, wave)
                else: loss_mel = F.l1_loss(y_mel, y_hat_mel)
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                loss_fm = feature_loss(fmap_r, fmap_g)
                harmonic_loss, tefs_loss, tsi_loss = combined_aux_loss(
                    wave, y_hat,n_mels=hps.data.n_mel_channels,sample_rate=hps.data.sampling_rate,
                    c_tefs=hps.train.get("c_tefs",0.),
                    c_hd=hps.train.get("c_hd",0.),
                    c_tsi=hps.train.get("c_tsi",0.),
                    n_fft=hps.data.filter_length,
                    hop_length=hps.data.hop_length,
                    win_length=hps.data.win_length,
                    eps=hps.train.eps,
                    fmin=hps.data.mel_fmin,
                    fmax=hps.data.mel_fmax,
                )
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                aux_loss = harmonic_loss + tefs_loss + tsi_loss
                loss_gen_all = balancer_g.on_train_batch_start(dict(
                    loss_gen=loss_gen,
                    loss_fm=loss_fm,
                    loss_mel=loss_mel,
                    loss_kl=loss_kl,
                    harmonic_loss=harmonic_loss,
                    tsi_loss=tsi_loss,
                    tefs_loss=tefs_loss,
                    ),input=y_hat)
                
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), gradient_clip_value, batch_size=hps.train.batch_size)
            scaler.step(optim_g)
            scaler.update()
        
        if rank == 0:
            if hps.train.log_interval>0 and global_step % hps.train.log_interval == 0: #tensorboard logging
                if hps.train.get("use_multiscale"): MultiscaleMelLoss.show_freqs()

                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9
                
                scalar_dict = {
                    "total/loss/all": loss_gen_all+loss_disc_all,
                    "total/loss/gen_all": loss_gen_all,
                    "total/loss/aux": aux_loss,
                    "total/loss/disc_all": loss_disc_all,
                    "total/loss/gen": loss_gen,
                    "total/loss/disc": loss_disc,
                    "total/loss/fm": loss_fm,
                    "total/loss/mel": loss_mel,
                    "total/loss/kl": loss_kl,
                    "aux/loss/harmonic": harmonic_loss,
                    "aux/loss/tefs": tefs_loss,
                    "aux/loss/tsi": tsi_loss,
                    "gradient/lr": lr,
                    "gradient/grad_norm_disc": grad_norm_d,
                    "gradient/grad_norm_gen": grad_norm_g,
                    "gradient/gradient_penalty": gradient_penalty,
                    **{f"loss/g/{i}": v for i, v in enumerate(losses_gen)},
                    **{f"loss/d/{i}": v for i, v in enumerate(losses_disc)},
                    **{f"balancer_g/weights/{k}": v for k, v in balancer_g.ema_weights.items()},
                    **{f"balancer_d/weights/{k}": v for k, v in balancer_d.ema_weights.items()},
                }

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "slice/diff^2": utils.plot_spectrogram_to_numpy((y_mel[0]-y_hat_mel[0]).pow(2).data.cpu().numpy(), cmap="hot")
                }

                with torch.no_grad():
                    if hasattr(net_g, "module"): inference = net_g.module.infer
                    else: inference = net_g.infer
                    if hps.if_f0: wave_gen = inference(phone, phone_lengths, pitch, pitchf, sid)[0][0, 0].data
                    else: wave_gen = inference(phone, phone_lengths, sid)[0][0, 0].data
                    
                audio_dict = {
                    "slice/wave_org": wave_orig[0][0],
                    "slice/wave_gen": wave_gen
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                    audios=audio_dict,
                    audio_sampling_rate=SR_MAP[hps.sample_rate]
                )
            if progress_tracker is not None:
                progress_tracker.update_batch(
                    epoch,
                    processed_step_idx,
                    steps_per_epoch,
                    global_step,
                    lr,
                    metrics={
                        "loss_gen_all": loss_gen_all,
                        "loss_disc_all": loss_disc_all,
                        "loss_mel": loss_mel,
                        "loss_kl": loss_kl,
                        "loss_fm": loss_fm,
                        "aux_loss": aux_loss,
                    },
                    source_batch_idx=batch_idx,
                )
        global_step += 1
    # /Run steps

    if hps.save_every_epoch>0 and (epoch % hps.save_every_epoch == 0) and rank == 0:
        msml_state = _serialize_multiscale_mel_state(hps)
        g_name = f"G_{epoch}.pth"
        d_name = f"D_{epoch}.pth"
        utils.save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, g_name),
            balancer=balancer_g.to_dict(),
            msml=msml_state
        )
        utils.save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, d_name),
            balancer=balancer_d.to_dict()
        )
        _prune_checkpoint_history(hps.model_dir, getattr(hps, "max_checkpoints", 1), logger_obj=logger)
        if hps.save_every_weights:
            ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
            save_name = f"{hps.name}_e{epoch}_s{global_step}"
            status = save_checkpoint(ckpt,save_name,epoch,hps)
            logger.info(f"saving ckpt {save_name}: {status}")

    if rank == 0:
        total_loss = balancer_g.weighted_ema_loss + balancer_d.weighted_ema_loss
        epoch_time_text, epoch_time_sec = epoch_recorder.snapshot()
        _log_epoch_summary(
            logger,
            epoch=epoch,
            hps=hps,
            global_step=global_step,
            lr=lr,
            epoch_time=epoch_time_text,
            metrics={
                "total_loss": total_loss,
                "loss_disc_all": loss_disc_all,
                "loss_gen_all": loss_gen_all,
                "loss_mel": loss_mel,
                "loss_kl": loss_kl,
                "loss_fm": loss_fm,
                "aux_loss": aux_loss,
            },
        )
        if progress_tracker is not None:
            progress_tracker.end_epoch(
                epoch,
                global_step,
                lr,
                epoch_time_sec,
                metrics={
                    "total_loss": total_loss,
                    "loss_disc_all": loss_disc_all,
                    "loss_gen_all": loss_gen_all,
                    "loss_mel": loss_mel,
                    "loss_kl": loss_kl,
                    "loss_fm": loss_fm,
                    "aux_loss": aux_loss,
                    "steps_per_epoch": steps_per_epoch,
                },
            )

        #sigmoid scaling of ema
        weights_decay = commons.sigmoid_value(global_step,total_steps=10000,start_value=.5, end_value=.999, midpoint=.2)
        balancer_g.on_epoch_end(weights_decay) 
        balancer_d.on_epoch_end(weights_decay)

        if loss_gen_all<least_loss:
            least_loss = loss_gen_all
            logger.info(f"Best checkpoint improved | gen_loss={least_loss:.3f}")

            if hps.save_best_model:
                if hasattr(net_g, "module"): ckpt = net_g.module.state_dict()
                else: ckpt = net_g.state_dict()
                
                best_model_name = f"{hps.name}_e{epoch}_s{global_step}_loss{least_loss:.0f}" if hps.save_every_weights else f"{hps.name}_loss{least_loss:2.0f}"
                status = save_checkpoint(ckpt,best_model_name,epoch,hps)
                logger.info(f"Saved best model | file={best_model_name}.pth | status={status}")
            
            with open(loss_file,"w") as f:
                json.dump(dict(least_loss=least_loss.item(),best_model_name=best_model_name,epoch=epoch,steps=global_step,
                                loss_weights = dict(**balancer_g.ema_weights,**balancer_d.ema_weights),
                                scalar_dict={
                                    "total/loss/all": commons.serialize_tensor(loss_gen_all+loss_disc_all),
                                    "total/loss/gen_all": commons.serialize_tensor(loss_gen_all),
                                    "total/loss/aux": commons.serialize_tensor(aux_loss),
                                    "total/loss/disc_all": commons.serialize_tensor(loss_disc_all),
                                    "total/loss/gen": commons.serialize_tensor(loss_gen),
                                    "total/loss/disc": commons.serialize_tensor(loss_disc),
                                    "total/loss/fm": commons.serialize_tensor(loss_fm),
                                    "total/loss/mel": commons.serialize_tensor(loss_mel),
                                    "total/loss/kl": commons.serialize_tensor(loss_kl),
                                    "aux/loss/harmonic": commons.serialize_tensor(harmonic_loss),
                                    "aux/loss/tefs": commons.serialize_tensor(tefs_loss),
                                    "aux/loss/tsi": commons.serialize_tensor(tsi_loss),
                                    "gradient/grad_norm_disc": commons.serialize_tensor(grad_norm_d),
                                    "gradient/grad_norm_gen": commons.serialize_tensor(grad_norm_g),
                                    "gradient/gradient_penalty": commons.serialize_tensor(gradient_penalty),
                                }),f,indent=2)

    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training complete")

        ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
        if hps.save_best_model and os.path.isfile(loss_file):
            with open(loss_file,"r") as f:
                data = json.load(f)
                best_model_name = data.get("best_model_name","")
                best_model_path = os.path.join(hps.model_dir,f"{best_model_name}.pth")
                if os.path.isfile(best_model_path):
                    shutil.copy(best_model_path,os.path.join(
                        os.path.dirname(hps.model_path),
                        f"{os.path.basename(hps.model_path).split('.')[0]}-lowest.pth"))

        status = save_checkpoint(ckpt,hps.name,epoch,hps,model_path=hps.model_path)
        logger.info(f"Saved final model | path={hps.model_path} | status={status}")
        if progress_tracker is not None:
            progress_tracker.finalize(
                "completed",
                "complete",
                model_path=hps.model_path,
                model_dir=hps.model_dir,
            )
        sleep(1)
        os._exit(0)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    hps = utils.get_hparams()
    train_model(hps)
