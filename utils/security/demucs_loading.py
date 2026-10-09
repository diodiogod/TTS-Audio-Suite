"""Read legacy Demucs metadata without unpickling executable model classes."""

import argparse
import importlib
from pathlib import Path


def load_demucs_package(*args, architecture_package, architecture_directory, loader=None, **kwargs):
    import torch

    if "pickle_module" in kwargs:
        raise ValueError("Custom pickle loaders are forbidden for Demucs checkpoints.")
    references = {}
    safe_globals = [argparse.Namespace]
    architectures = [("demucs", "Demucs"), ("model", "Demucs"),
                     ("hdemucs", "HDemucs"), ("htdemucs", "HTDemucs"),
                     ("tasnet_v2", "ConvTasNet")]
    prefixes = {"", "demucs.", "lib.uvr5_pack.demucs.",
                "engines.rvc.impl.lib.uvr5_pack.demucs.", architecture_package + "."}
    for module, name in architectures:
        # Keep references as inert tokens. REDUCE must never construct a model.
        def token(*_args, **_kwargs):
            raise ValueError("Checkpoint architecture references cannot construct pickle objects.")
        references[token] = (module, name)
        safe_globals.extend((token, f"{prefix}{module}.{name}") for prefix in prefixes)
    kwargs["weights_only"] = True
    with torch.serialization.safe_globals(safe_globals):
        package = (loader or torch.load)(*args, **kwargs)
    if not isinstance(package, dict) or package.get("klass") not in references:
        raise ValueError("Unsupported Demucs architecture metadata.")
    module, name = references[package["klass"]]
    local_module = importlib.import_module("." + module, architecture_package) if architecture_package else importlib.import_module(module)
    expected = Path(architecture_directory, module + ".py").resolve()
    if Path(local_module.__file__).resolve() != expected:
        raise ValueError("Demucs architecture must come from its installed runtime.")
    package["klass"] = getattr(local_module, name)
    return package
