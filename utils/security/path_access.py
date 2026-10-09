"""Contain workflow-supplied paths in ComfyUI's configured data folders."""

import os
import re
from functools import wraps
from inspect import signature
from pathlib import Path


def data_roots(*, models=True):
    import folder_paths

    roots = [folder_paths.get_input_directory(), folder_paths.get_output_directory(), folder_paths.get_temp_directory()]
    if models:
        roots.append(folder_paths.models_dir)
        for name, (paths, _extensions) in getattr(folder_paths, "folder_names_and_paths", {}).items():
            if name == "custom_nodes":
                continue
            roots.extend(paths)
        roots.append(Path(__file__).resolve().parents[2] / "voices_examples")
    return [Path(root).resolve() for root in roots if root]


def contained_path(path, roots):
    candidate = Path(path).expanduser().resolve()
    if not any(candidate.is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError("File path is outside the permitted ComfyUI data folders. Upload the file or move it into ComfyUI input/; model and voice folders can be registered in extra_model_paths.yaml.")
    return str(candidate)


def allowed_path(path, *, models=True):
    return contained_path(path, data_roots(models=models))


def resolve_input_path(value, *, datasets=False, models=True):
    import folder_paths

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("A file path is required.")
    if os.path.isabs(raw):
        return allowed_path(raw, models=models)
    # Honor ComfyUI's [input], [output], and [temp] annotations.
    if raw.endswith((" [input]", " [output]", " [temp]")):
        return allowed_path(folder_paths.get_annotated_filepath(raw), models=models)
    candidates = [Path(folder_paths.get_input_directory()) / raw]
    if datasets:
        candidates.append(Path(folder_paths.get_input_directory()) / "datasets" / raw)
    for candidate in candidates:
        checked = allowed_path(candidate, models=models)
        if Path(checked).exists():
            return checked
    return allowed_path(candidates[0], models=models)


def filename_component(value):
    value = str(value or "0")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("Invalid node ID for an audio cache filename.")
    return value


def child_path(root, *parts):
    return contained_path(Path(root).joinpath(*parts), [root])


def validate_model_identifier(value):
    """Keep model selections as selections; validate explicit filesystem paths."""
    if not value:
        return value
    raw = os.fspath(value)
    path = raw[6:] if raw.startswith("local:") else raw
    if os.path.isabs(path):
        allowed_path(path)
    elif Path(path).drive or ".." in path.replace("\\", "/").split("/"):
        raise ValueError("Model selections cannot contain parent traversal or relative drive paths.")
    return value


def validate_model_paths(resolve):
    """Apply the same checks to the suite's model downloaders without duplicating them."""
    parameters = signature(resolve)

    @wraps(resolve)
    def checked(*args, **kwargs):
        values = parameters.bind_partial(*args, **kwargs).arguments
        for name in ("model_identifier", "identifier", "selection"):
            if name in values:
                validate_model_identifier(values[name])

        def check_result(value):
            if isinstance(value, dict):
                return {key: check_result(item) for key, item in value.items()}
            if isinstance(value, (str, os.PathLike)):
                return allowed_path(value)
            return value

        return check_result(resolve(*args, **kwargs))

    return checked
