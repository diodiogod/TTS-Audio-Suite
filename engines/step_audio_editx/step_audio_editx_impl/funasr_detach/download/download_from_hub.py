import os
import json
import threading
from omegaconf import OmegaConf

from funasr_detach.download.name_maps_from_hub import name_maps_ms, name_maps_hf

# Global cache for downloaded models to avoid repeated downloads
# Key: (repo_id, model_revision, model_hub)
# Value: repo_cache_dir
_model_cache = {}
_cache_lock = threading.Lock()


def download_model(**kwargs):
    model_hub = kwargs.get("model_hub", "ms")
    model_or_path = kwargs.get("model")
    repo_path = kwargs.get("repo_path", "")

    # Handle name mapping based on model_hub
    if model_hub == "ms" and model_or_path in name_maps_ms:
        model_or_path = name_maps_ms[model_or_path]
    elif model_hub == "hf" and model_or_path in name_maps_hf:
        model_or_path = name_maps_hf[model_or_path]

    model_revision = kwargs.get("model_revision")

    # Download model if it doesn't exist locally
    if not os.path.exists(model_or_path):
        if model_hub == "local":
            # For local models, the path should already exist
            raise FileNotFoundError(f"Local model path does not exist: {model_or_path}")
        elif model_hub in ["ms", "hf"]:
            repo_path, model_or_path = get_or_download_model_dir(
                model_or_path,
                model_revision,
                is_training=kwargs.get("is_training"),
                check_latest=kwargs.get("kwargs", True),
                model_hub=model_hub,
            )
        else:
            raise ValueError(f"Unsupported model_hub: {model_hub}")

    # Model path used silently (printed every run)
    kwargs["model_path"] = model_or_path
    kwargs["repo_path"] = repo_path

    # Common logic for processing configuration files (same for all model hubs)
    if os.path.exists(os.path.join(model_or_path, "configuration.json")):
        with open(
            os.path.join(model_or_path, "configuration.json"), "r", encoding="utf-8"
        ) as f:
            conf_json = json.load(f)
            cfg = {}
            cfg = add_file_root_path(model_or_path, conf_json["file_path_metas"], cfg)
            cfg.update(kwargs)
            config = OmegaConf.load(cfg["config"])
            kwargs = OmegaConf.merge(config, cfg)
        kwargs["model"] = config["model"]
    elif os.path.exists(os.path.join(model_or_path, "config.yaml")) and (
        os.path.exists(os.path.join(model_or_path, "model.pt")) or
        os.path.exists(os.path.join(model_or_path, "model.pb"))
    ):
        config = OmegaConf.load(os.path.join(model_or_path, "config.yaml"))
        kwargs = OmegaConf.merge(config, kwargs)

        # Convert to dict to allow modifications, then convert back to OmegaConf
        kwargs_dict = OmegaConf.to_container(kwargs, resolve=True)

        init_param = os.path.join(model_or_path, "model.pb")
        kwargs_dict["init_param"] = init_param
        if os.path.exists(os.path.join(model_or_path, "tokens.txt")):
            kwargs_dict["tokenizer_conf"]["token_list"] = os.path.join(
                model_or_path, "tokens.txt"
            )
        if os.path.exists(os.path.join(model_or_path, "tokens.json")):
            kwargs_dict["tokenizer_conf"]["token_list"] = os.path.join(
                model_or_path, "tokens.json"
            )
        if os.path.exists(os.path.join(model_or_path, "seg_dict")):
            kwargs_dict["tokenizer_conf"]["seg_dict"] = os.path.join(
                model_or_path, "seg_dict"
            )
        if os.path.exists(os.path.join(model_or_path, "bpe.model")):
            kwargs_dict["tokenizer_conf"]["bpemodel"] = os.path.join(
                model_or_path, "bpe.model"
            )
        kwargs_dict["model"] = config["model"]
        if os.path.exists(os.path.join(model_or_path, "am.mvn")):
            kwargs_dict["frontend_conf"]["cmvn_file"] = os.path.join(model_or_path, "am.mvn")
        if os.path.exists(os.path.join(model_or_path, "jieba_usr_dict")):
            kwargs_dict["jieba_usr_dict"] = os.path.join(model_or_path, "jieba_usr_dict")

        return kwargs_dict


def add_file_root_path(model_or_path: str, file_path_metas: dict, cfg={}):

    if isinstance(file_path_metas, dict):
        for k, v in file_path_metas.items():
            if isinstance(v, str):
                p = os.path.join(model_or_path, v)
                if os.path.exists(p):
                    cfg[k] = p
            elif isinstance(v, dict):
                if k not in cfg:
                    cfg[k] = {}
                add_file_root_path(model_or_path, v, cfg[k])

    return cfg


def get_or_download_model_dir(
    model,
    model_revision=None,
    is_training=False,
    check_latest=True,
    model_hub="ms",
):
    """Get local model directory or download model if necessary.

    Args:
        model (str): model id or path to local model directory.
                    For HF subfolders, use format: "repo_id/subfolder_path"
        model_revision  (str, optional): model version number.
        is_training (bool): Whether this is for training
        check_latest (bool): Whether to check for latest version
        model_hub (str): Model hub type ("ms" for ModelScope, "hf" for HuggingFace)
    """
    # Extract repo_id for caching (handle subfolder case)
    if "/" in model and len(model.split("/")) > 2:
        parts = model.split("/")
        repo_id = "/".join(parts[:2])  # e.g., "organization/repo" or "stepfun-ai/Step-Audio-EditX"
        subfolder = "/".join(parts[2:])  # e.g., "subfolder/model"
    else:
        repo_id = model
        subfolder = None

    # Create cache key
    cache_key = (repo_id, model_revision, model_hub)

    # Check cache first
    with _cache_lock:
        if cache_key in _model_cache:
            cached_repo_dir = _model_cache[cache_key]
            # Using cached model silently (happens every run)

            # For subfolder case, construct the model_cache_dir from cached repo
            if subfolder:
                model_cache_dir = os.path.join(cached_repo_dir, subfolder)
                if not os.path.exists(model_cache_dir):
                    raise FileNotFoundError(f"Subfolder {subfolder} not found in cached repo {repo_id}")
            else:
                model_cache_dir = cached_repo_dir

            return cached_repo_dir, model_cache_dir

    # Cache miss, need to download
    if model_hub == "ms":
        # ModelScope download
        from modelscope.hub.snapshot_download import snapshot_download
        from modelscope.utils.constant import Invoke, ThirdParty

        key = Invoke.LOCAL_TRAINER if is_training else Invoke.PIPELINE

        # Download the repo (use repo_id, not the full model path with subfolder)
        repo_cache_dir = snapshot_download(
            repo_id,
            revision=model_revision,
            user_agent={Invoke.KEY: key, ThirdParty.KEY: "funasr"},
        )
        repo_cache_dir = normalize_cache_path(repo_cache_dir)

        # Construct model_cache_dir
        if subfolder:
            model_cache_dir = os.path.join(repo_cache_dir, subfolder)
            if not os.path.exists(model_cache_dir):
                raise FileNotFoundError(f"Subfolder {subfolder} not found in downloaded repo {repo_id}")
        else:
            model_cache_dir = normalize_cache_path(repo_cache_dir)

    elif model_hub == "hf":
        # HuggingFace download
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloading from HuggingFace. "
                "Please install it with: pip install huggingface_hub"
            )

        # Download the repo (use repo_id, not the full model path with subfolder)
        repo_cache_dir = snapshot_download(
            repo_id=repo_id,
            revision=model_revision,
            allow_patterns=None,  # Download all files to ensure resource files are available
        )
        repo_cache_dir = normalize_cache_path(repo_cache_dir)

        # Construct model_cache_dir
        if subfolder:
            model_cache_dir = os.path.join(repo_cache_dir, subfolder)
            if not os.path.exists(model_cache_dir):
                raise FileNotFoundError(f"Subfolder {subfolder} not found in downloaded repo {repo_id}")
        else:
            model_cache_dir = normalize_cache_path(repo_cache_dir)
    else:
        raise ValueError(f"Unsupported model_hub: {model_hub}")

    # Cache the result before returning
    with _cache_lock:
        _model_cache[cache_key] = repo_cache_dir

    print(f"Model downloaded to: {model_cache_dir}")
    return repo_cache_dir, model_cache_dir

def normalize_cache_path(cache_path):
    """Normalize cache path to ensure consistent format with snapshots/{commit_id}."""
    # Check if the cache_path directory contains a snapshots folder
    snapshots_dir = os.path.join(cache_path, "snapshots")
    if os.path.exists(snapshots_dir) and os.path.isdir(snapshots_dir):
        # Find the commit_id subdirectory in snapshots
        try:
            snapshot_items = os.listdir(snapshots_dir)
            # Look for the first directory (should be the commit_id)
            for item in snapshot_items:
                item_path = os.path.join(snapshots_dir, item)
                if os.path.isdir(item_path):
                    # Found commit_id directory, return the full path
                    return os.path.join(cache_path, "snapshots", item)
        except OSError:
            pass

    # If no snapshots directory found or error occurred, return original path
    return cache_path

