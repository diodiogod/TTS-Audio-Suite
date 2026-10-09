"""Load tensor checkpoints and plain dictionaries without executable pickle globals."""

import pickle


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Executable pickle global is forbidden: {module}.{name}")


def load_data_pickle(handle):
    return DataOnlyUnpickler(handle).load()


class NumpyDataUnpickler(DataOnlyUnpickler):
    def find_class(self, module, name):
        import numpy as np

        core = np._core if hasattr(np, "_core") else np.core
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): core.multiarray.scalar,
        }
        if (module, name) in allowed:
            return allowed[module, name]
        return super().find_class(module, name)


def load_numpy_data(path):
    """Keep .npy/.dac data dictionaries while refusing arbitrary pickle globals."""
    import numpy as np

    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        shape, _fortran, dtype = np.lib.format._read_array_header(handle, version)
        if not dtype.hasobject:
            handle.seek(0)
            return np.load(handle, allow_pickle=False)
        result = NumpyDataUnpickler(handle).load()
        if not isinstance(result, np.ndarray) or result.shape != shape or result.dtype != dtype:
            raise ValueError("NumPy data does not match its array header.")
        return result


def load_torch_checkpoint(*args, **kwargs):
    import numpy as np
    import torch

    if kwargs.get("weights_only") is False or "pickle_module" in kwargs:
        raise ValueError("Checkpoint loading must use weights_only=True.")
    kwargs["weights_only"] = True
    # Dots latent statistics contain NumPy arrays. This fixed, scoped allowlist
    # supports those data values without accepting arbitrary checkpoint classes.
    numpy_core = np._core if hasattr(np, "_core") else np.core
    safe_types = [
        numpy_core.multiarray._reconstruct,
        (numpy_core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        type(np.dtype("float32")),
        type(np.dtype("float64")),
    ]
    with torch.serialization.safe_globals(safe_types):
        return torch.load(*args, **kwargs)


class RestrictedLoadModule:
    """Restrict a dependency's own module reference without patching Python globally."""

    def __init__(self, module, load=None, **overrides):
        self._module = module
        self._overrides = overrides
        if load is not None:
            self._overrides["load"] = load

    def __getattr__(self, name):
        return self._overrides[name] if name in self._overrides else getattr(self._module, name)


def load_dependency_checkpoint(*args, **kwargs):
    # Override a dependency's explicit unrestricted flag in this integration.
    kwargs["weights_only"] = True
    return load_torch_checkpoint(*args, **kwargs)
