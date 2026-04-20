"""PyTorch autograd wrappers for the GSCP CUDA 2D Gaussian rasterizer.

Provides:
- ``GaussianRasterize2D`` — RS parameterization (rotation + scaling)
- ``GaussianRasterize2DCholesky`` — Cholesky parameterization (L @ L^T)

Both are :class:`torch.autograd.Function` subclasses that serve as drop-in
replacements for the corresponding PyTorch scatter-add renderers.
"""

from __future__ import annotations

import torch

from gscp.cuda import _C


class GaussianRasterize2D(torch.autograd.Function):
    """Custom autograd function wrapping the CUDA tile-based 2D rasterizer.

    Forward:  (xy, scaling, rotation, weight, H, W, max_patch_radius) -> [2, H, W]
    Backward: dL_d[2,H,W] -> (dL_dxy, dL_dscaling, dL_drotation, dL_dweight)
    """

    @staticmethod
    def forward(
        ctx,
        xy: torch.Tensor,  # [N, 2]
        scaling: torch.Tensor,  # [N, 2]
        rotation: torch.Tensor,  # [N, 1]
        weight: torch.Tensor,  # [N, 2]
        H: int,
        W: int,
        max_patch_radius: int,
        min_scale: float = 0.5,
    ) -> torch.Tensor:
        num_rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer = (
            _C.rasterize_forward(
                xy.contiguous(),
                scaling.contiguous(),
                rotation.contiguous(),
                weight.contiguous(),
                H,
                W,
                max_patch_radius,
                min_scale,
                False,  # debug
            )
        )

        ctx.save_for_backward(
            xy, scaling, rotation, weight, radii, geomBuffer, binningBuffer, imgBuffer
        )
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.max_patch_radius = max_patch_radius
        ctx.min_scale = min_scale

        return out_color  # [2, H, W]

    @staticmethod
    def backward(ctx, dL_dout: torch.Tensor):
        xy, scaling, rotation, weight, radii, geomBuffer, binningBuffer, imgBuffer = (
            ctx.saved_tensors
        )

        dL_dxy, dL_dscaling, dL_drotation, dL_dweight = _C.rasterize_backward(
            xy,
            scaling,
            rotation,
            weight,
            radii,
            ctx.H,
            ctx.W,
            ctx.max_patch_radius,
            ctx.min_scale,
            ctx.num_rendered,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            dL_dout.contiguous(),
            False,  # debug
        )

        # Gradients: (xy, scaling, rotation, weight, H, W, max_patch_radius, min_scale)
        return dL_dxy, dL_dscaling, dL_drotation, dL_dweight, None, None, None, None


class GaussianRasterize2DCholesky(torch.autograd.Function):
    """Custom autograd function for Cholesky-parameterized 2D Gaussian rasterizer.

    Forward:  (xy, log_L_diag, L_offdiag, weight, H, W, max_patch_radius, min_scale) -> [2, H, W]
    Backward: dL_d[2,H,W] -> (dL_dxy, dL_dlog_L_diag, dL_dL_offdiag, dL_dweight)
    """

    @staticmethod
    def forward(
        ctx,
        xy: torch.Tensor,  # [N, 2]
        log_L_diag: torch.Tensor,  # [N, 2]
        L_offdiag: torch.Tensor,  # [N]
        weight: torch.Tensor,  # [N, 2]
        H: int,
        W: int,
        max_patch_radius: int,
        min_scale: float = 0.5,
    ) -> torch.Tensor:
        num_rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer = (
            _C.rasterize_forward_cholesky(
                xy.contiguous(),
                log_L_diag.contiguous(),
                L_offdiag.contiguous(),
                weight.contiguous(),
                H,
                W,
                max_patch_radius,
                min_scale,
                False,  # debug
            )
        )

        ctx.save_for_backward(
            xy, log_L_diag, L_offdiag, weight, radii,
            geomBuffer, binningBuffer, imgBuffer,
        )
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.max_patch_radius = max_patch_radius
        ctx.min_scale = min_scale

        return out_color  # [2, H, W]

    @staticmethod
    def backward(ctx, dL_dout: torch.Tensor):
        (
            xy, log_L_diag, L_offdiag, weight, radii,
            geomBuffer, binningBuffer, imgBuffer,
        ) = ctx.saved_tensors

        dL_dxy, dL_dlog_L_diag, dL_dL_offdiag, dL_dweight = (
            _C.rasterize_backward_cholesky(
                xy,
                log_L_diag,
                L_offdiag,
                weight,
                radii,
                ctx.H,
                ctx.W,
                ctx.max_patch_radius,
                ctx.min_scale,
                ctx.num_rendered,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                dL_dout.contiguous(),
                False,  # debug
            )
        )

        # Gradients: (xy, log_L_diag, L_offdiag, weight, H, W, max_patch_radius, min_scale)
        return dL_dxy, dL_dlog_L_diag, dL_dL_offdiag, dL_dweight, None, None, None, None
