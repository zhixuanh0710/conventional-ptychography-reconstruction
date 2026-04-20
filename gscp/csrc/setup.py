"""Ahead-of-time build script for the GSCP CUDA rasterizer.

Usage:
    pip install gscp/csrc/
    # or
    cd gscp/csrc && pip install .

Supports both Linux (gcc) and Windows (MSVC).
"""

import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_dir = os.path.dirname(os.path.abspath(__file__))

# Platform-specific C++ compiler flags
if sys.platform == "win32":
    cxx_flags = ["/O2"]
else:
    cxx_flags = ["-O3"]

# NVCC flags — let PyTorch auto-detect GPU arch via TORCH_CUDA_ARCH_LIST
# or visible GPUs. No hardcoded -gencode for portability.
nvcc_flags = ["-O3", "--use_fast_math"]

setup(
    name="gscp_rasterizer",
    ext_modules=[
        CUDAExtension(
            name="gscp_rasterizer._C",
            sources=[
                os.path.join(_src_dir, "ext.cpp"),
                # rasterize_gaussians is pure C++ (no CUDA kernels), compiled
                # as .cpp to avoid nvcc+MSVC 14.44 header incompatibility.
                os.path.join(_src_dir, "rasterize_gaussians.cpp"),
                os.path.join(_src_dir, "gscp_rasterizer", "forward.cu"),
                os.path.join(_src_dir, "gscp_rasterizer", "backward.cu"),
                os.path.join(_src_dir, "gscp_rasterizer", "rasterizer_impl.cu"),
            ],
            include_dirs=[
                os.path.join(_src_dir, "gscp_rasterizer"),
                _src_dir,
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
