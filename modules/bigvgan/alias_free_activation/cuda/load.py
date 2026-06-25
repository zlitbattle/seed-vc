# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import pathlib

import torch
from torch.utils import cpp_extension


def load():
    configure_cuda_arch_list()

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
                "--use_fast_math",
            ]
            + extra_cuda_flags,
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


def configure_cuda_arch_list():
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    arch_list = f"{major}.{minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    print(
        "Building BigVGAN fused CUDA kernel for current GPU architecture: "
        f"TORCH_CUDA_ARCH_LIST={arch_list}"
    )


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
