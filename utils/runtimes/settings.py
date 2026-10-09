"""Local installation preferences shared by ComfyUI and install.py."""

import json
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _comfy_root():
    folder_paths = sys.modules.get("folder_paths")
    comfy_root = os.environ.get("COMFYUI_FOLDERS_BASE_PATH") or os.environ.get("COMFYUI_PATH")
    if not comfy_root and getattr(folder_paths, "__file__", None):
        comfy_root = str(Path(folder_paths.__file__).parent)
    comfy_root = comfy_root or getattr(folder_paths, "base_path", None)
    if not comfy_root:
        package = Path(os.path.abspath(__file__)).parents[2]
        if package.parent.name == "custom_nodes":
            comfy_root = str(package.parent.parent)
        elif (Path.cwd() / "folder_paths.py").is_file():
            comfy_root = str(Path.cwd())
    if not comfy_root:
        raise RuntimeError("Cannot locate ComfyUI. Set COMFYUI_PATH when running install.py outside ComfyUI.")
    return Path(comfy_root)


def location_file():
    # Keep the pointer outside the package too, so a Manager reinstall can find
    # the preference stored under a custom --user-directory.
    return _comfy_root() / "user" / "__tts_audio_suite" / "runtime_settings_location.json"


def settings_path():
    # The live server knows --user-directory. Remember it for Manager's separate
    # installer process, which does not receive ComfyUI's command-line arguments.
    folder_paths = sys.modules.get("folder_paths")
    if folder_paths is not None and hasattr(folder_paths, "get_system_user_directory"):
        return Path(folder_paths.get_system_user_directory("tts_audio_suite")) / "runtime_settings.json"
    user_dir = os.environ.get("TTS_AUDIO_SUITE_USER_DIRECTORY")
    if user_dir:
        return Path(user_dir) / "__tts_audio_suite" / "runtime_settings.json"
    pointer = location_file()
    if pointer.is_file():
        location = json.loads(pointer.read_text(encoding="utf-8"))
        return Path(location["user_directory"]) / "__tts_audio_suite" / "runtime_settings.json"
    return _comfy_root() / "user" / "__tts_audio_suite" / "runtime_settings.json"


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def remember_user_directory():
    path = settings_path()
    _write_json(location_file(), {"user_directory": str(path.parent.parent.absolute())})


def validate_settings(value):
    if not isinstance(value, dict) or set(value) != {"install_shared_runtime"}:
        raise ValueError("Expected only install_shared_runtime in runtime settings.")
    if type(value["install_shared_runtime"]) is not bool:
        raise ValueError("install_shared_runtime must be true or false.")
    return value


def read_settings():
    path = settings_path()
    if not path.is_file():
        return {"install_shared_runtime": True}
    return validate_settings(json.loads(path.read_text(encoding="utf-8-sig")))


def save_settings(value):
    value = validate_settings(value)
    _write_json(settings_path(), value)
    remember_user_directory()
    return value
