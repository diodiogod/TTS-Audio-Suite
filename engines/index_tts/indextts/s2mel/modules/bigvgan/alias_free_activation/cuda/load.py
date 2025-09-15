# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import pathlib
import subprocess

from torch.utils import cpp_extension

"""
Setting this param to a list has a problem of generating different compilation commands (with diferent order of architectures) and leading to recompilation of fused kernels. 
Set it to empty stringo avoid recompilation and assign arch flags explicity in extra_cuda_cflags below
"""
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load():
    try:
        # Check if cuda 11 is installed for compute capability 8.0
        cc_flag = []
        _, bare_metal_major, _ = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
        if int(bare_metal_major) >= 11:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_80,code=sm_80")

        # Build path
        srcpath = pathlib.Path(__file__).parent.absolute()
        buildpath = srcpath / "build"
        _create_build_dir(buildpath)

        # Helper function to build the kernels.
        def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
            return cpp_extension.load(
                name=name,
                sources=sources,
                build_directory=buildpath,
                extra_cflags=[
                    "-O3",
                ],
                extra_cuda_cflags=[
                    "-O3",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                    "--use_fast_math",
                ]
                + extra_cuda_flags
                + cc_flag,
                verbose=True,
            )

        extra_cuda_flags = [
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]

        sources = [
            srcpath / "anti_alias_activation.cpp",
            srcpath / "anti_alias_activation_cuda.cu",
        ]
        anti_alias_activation_cuda = _cpp_extention_load_helper(
            "anti_alias_activation_cuda", sources, extra_cuda_flags
        )

        return anti_alias_activation_cuda
    
    except Exception as e:
        print(f"WARNING: Failed to compile CUDA kernels: {e}")
        print("INFO: BigVGAN will use slower PyTorch fallback operations")
        raise RuntimeError(f"CUDA kernel compilation failed: {e}")


def _get_cuda_bare_metal_version(cuda_dir):
    # Handle Windows vs Linux paths for nvcc and strip any whitespace
    cuda_dir = cuda_dir.strip() if cuda_dir else ""
    nvcc_path = os.path.join(cuda_dir, "bin", "nvcc.exe" if os.name == "nt" else "nvcc")
    raw_output = subprocess.check_output(
        [nvcc_path, "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
