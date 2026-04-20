"""Fourier-domain utilities shared across models."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def fourier_upsample(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Upsample a complex 2D field by zero-padding in Fourier domain.

    Handles intermittent cuFFT failures on some GPUs (e.g. RTX 5090) via
    retry with cache clear and CPU fallback.

    Args:
        x: Complex tensor ``[B, C, H, W]``.
        scale_factor: Integer upsampling factor.

    Returns:
        Complex tensor ``[B, C, H*sf, W*sf]``.
    """
    B, C, H, W = x.shape
    new_H, new_W = H * scale_factor, W * scale_factor
    pad_h = (new_H - H) // 2
    pad_w = (new_W - W) // 2

    def _do(t: torch.Tensor) -> torch.Tensor:
        t_fft = torch.fft.fftshift(torch.fft.fft2(t, dim=(-2, -1)), dim=(-2, -1))
        t_fft_padded = F.pad(
            t_fft,
            (pad_w, new_W - W - pad_w, pad_h, new_H - H - pad_h),
            mode="constant",
            value=0,
        )
        t_up = torch.fft.ifft2(
            torch.fft.ifftshift(t_fft_padded, dim=(-2, -1)), dim=(-2, -1)
        )
        return t_up * (scale_factor**2)

    if not x.is_cuda:
        return _do(x)

    try:
        return _do(x)
    except RuntimeError as e:
        if "cuFFT" not in str(e) and "CUFFT" not in str(e):
            raise
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.backends.cuda.cufft_plan_cache.clear()
            return _do(x)
        except RuntimeError:
            logger.warning("cuFFT failed after retry, falling back to CPU: %s", e)
            return _do(x.cpu()).to(device=x.device, dtype=x.dtype)
