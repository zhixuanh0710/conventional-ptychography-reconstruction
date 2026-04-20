"""GSCP CUDA rasterizer — JIT compilation loader with fallback.

Tries to load the pre-built extension first, then falls back to JIT compilation.
If neither works, sets CUDA_AVAILABLE = False and the model falls back to PyTorch.

Supports both Linux (gcc) and Windows (MSVC) build environments.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

logger = logging.getLogger(__name__)

_C = None
CUDA_AVAILABLE = False


def _get_compile_flags() -> tuple[list[str], list[str]]:
    """Return (cxx_flags, nvcc_flags) appropriate for the current platform."""
    if sys.platform == "win32":
        cxx_flags = ["/O2"]
        nvcc_flags = ["-O3", "--use_fast_math"]
    else:
        cxx_flags = ["-O3"]
        nvcc_flags = ["-O3", "--use_fast_math"]
    return cxx_flags, nvcc_flags


# Try 1: load pre-built extension (from `pip install gscp/csrc/`)
try:
    from gscp_rasterizer import _C as _prebuilt  # type: ignore[import-not-found]

    _C = _prebuilt
    CUDA_AVAILABLE = True
    logger.info("Loaded pre-built GSCP CUDA rasterizer")
except ImportError:
    pass

# Ensure MSVC cl.exe is discoverable for JIT compilation on Windows.
if sys.platform == "win32" and _C is None:
    _msvc_search_roots = [
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Microsoft Visual Studio"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Microsoft Visual Studio"),
    ]
    for _root in _msvc_search_roots:
        _vc_tools = os.path.join(_root, "2022", "BuildTools", "VC", "Tools", "MSVC")
        if os.path.isdir(_vc_tools):
            for _ver in sorted(os.listdir(_vc_tools), reverse=True):
                _cl_dir = os.path.join(_vc_tools, _ver, "bin", "Hostx64", "x64")
                if os.path.isfile(os.path.join(_cl_dir, "cl.exe")):
                    if _cl_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = _cl_dir + os.pathsep + os.environ["PATH"]
                        logger.debug("Auto-detected MSVC cl.exe: %s", _cl_dir)
                    break

# Try 2: JIT compilation
if _C is None:
    try:
        from torch.utils.cpp_extension import load

        _src_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "csrc"
        )
        _cuda_sources = [
            os.path.join(_src_dir, "ext.cpp"),
            # rasterize_gaussians is pure C++ (no CUDA kernels), compiled
            # as .cpp to avoid nvcc+MSVC 14.44 header incompatibility.
            os.path.join(_src_dir, "rasterize_gaussians.cpp"),
            os.path.join(_src_dir, "gscp_rasterizer", "forward.cu"),
            os.path.join(_src_dir, "gscp_rasterizer", "backward.cu"),
            os.path.join(_src_dir, "gscp_rasterizer", "rasterizer_impl.cu"),
        ]
        _include_dirs = [
            os.path.join(_src_dir, "gscp_rasterizer"),
            _src_dir,
        ]
        _cxx_flags, _nvcc_flags = _get_compile_flags()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="_get_vc_env is private",
                category=UserWarning,
            )
            _C = load(
                name="gscp_rasterizer",
                sources=_cuda_sources,
                extra_include_paths=_include_dirs,
                extra_cuda_cflags=_nvcc_flags,
                extra_cflags=_cxx_flags,
                verbose=False,
            )
        CUDA_AVAILABLE = True
        logger.info("JIT-compiled GSCP CUDA rasterizer")
    except Exception as e:
        logger.warning("CUDA rasterizer unavailable, falling back to PyTorch: %s", e)
