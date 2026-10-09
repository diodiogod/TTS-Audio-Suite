"""Restrict audio-separator's module references as its architectures load."""

from functools import partial
import importlib
from pathlib import Path
import sys

from .data_loading import RestrictedLoadModule, load_dependency_checkpoint
from .demucs_loading import load_demucs_package


def restrict_audio_separator_loaders():
    separator = importlib.import_module("audio_separator.separator.separator")
    if getattr(separator, "_tts_restricted_loaders", False):
        return
    original_import = separator.importlib.import_module
    base = "audio_separator.separator."
    demucs = base + "uvr_lib_v5.demucs"
    plain_modules = [base + "architectures.vr_separator", base + "architectures.mdxc_separator",
                     base + "uvr_lib_v5.mdxnet", demucs + ".pretrained"]

    def restricted_import(*args, **kwargs):
        result = original_import(*args, **kwargs)
        # Architecture imports finish before their constructors read checkpoints.
        # Patch only these dependency modules, never the shared torch/importlib objects.
        for name in plain_modules + [demucs + ".states", demucs + ".repo"]:
            module = sys.modules.get(name)
            if module is None or not hasattr(module, "torch"):
                continue
            torch_module = importlib.import_module("torch")
            if name in (demucs + ".states", demucs + ".repo"):
                read = partial(load_demucs_package, architecture_package=demucs,
                               architecture_directory=Path(module.__file__).parent)
                read_url = partial(read, loader=torch_module.hub.load_state_dict_from_url)
            else:
                read = load_dependency_checkpoint
                def read_url(*url_args, **url_kwargs):
                    url_kwargs["weights_only"] = True
                    return torch_module.hub.load_state_dict_from_url(*url_args, **url_kwargs)
            hub = RestrictedLoadModule(torch_module.hub, load_state_dict_from_url=read_url)
            module.torch = RestrictedLoadModule(torch_module, read, hub=hub)
        return result

    separator.importlib = RestrictedLoadModule(separator.importlib, import_module=restricted_import)
    separator._tts_restricted_loaders = True
