"""Abstract base class for 2D Gaussian field models.

Provides the parameterization-agnostic infrastructure shared by all Gaussian
splatting representations: position/weight/opacity management, Fourier
upsampling, forward-pass orchestration, and the adaptive densification
framework (clone / split / prune).

Concrete subclasses implement geometry-specific behaviour:

* **Rotation-Scaling (RS)** — :class:`~gscp.models.gaussian_field.GaussianFieldModel`
* **Cholesky** — planned, see ``sample_code/network_GS_cholesky.py``

Subclass contract
-----------------
1. Define :attr:`_NAME_TO_ATTR` as a class-level ``dict[str, str]`` mapping
   optimizer group names to model attribute names.  Must include the common
   keys ``"xy"``, ``"weight"``, ``"opacity"`` plus any geometry-specific keys.
2. Implement all ``@abstractmethod``-decorated methods (see list below).
3. Store ``_weight`` as ``nn.Parameter`` of shape ``[N, 2]`` (real, imag).
   This is a base-class contract — all subclasses must use this layout.
"""

import abc
import logging
import math
from typing import Any

import torch
import torch.nn as nn

from gscp.utils.fourier import fourier_upsample as _fourier_upsample

logger = logging.getLogger(__name__)


class GaussianFieldCore(abc.ABC, nn.Module):
    """Abstract base for 2D Gaussian Splatting field models.

    See module docstring for the full subclass contract.

    Args:
        output_H: Full-resolution output height.
        output_W: Full-resolution output width.
        downsample_factor: Splatting canvas is ``(H // ds, W // ds)``.
        num_initial_gaussians: Number of Gaussians at init.
        object_recovery_initial: Optional complex ``[H, W]`` seed field.
        densify_grad_threshold: Min gradient norm for densification.
        densify_scale_threshold: Fraction of ``max(H, W)`` for clone/split.
        densify_interval: Steps between densification attempts.
        densify_until_step: Disable densification after this step.
        prune_weight_threshold: Min effective weight to survive pruning.
        opacity_aware_pruning: Use ``weight * opacity`` for pruning.
        max_gaussians: Hard upper bound on Gaussian count.
        render_batch_size: Legacy hint for PyTorch renderer.
        max_patch_radius: Max pixel radius of splatting patch.
        init_scale: Initial sigma in downsampled pixels.
        min_scale: Minimum activated sigma in pixels (clamped >= 0.05).
        position_lr_scale: LR multiplier for positions.
        weight_lr_scale: LR multiplier for weights and opacity.
        **kwargs: Forwarded to subclass (e.g. ``scaling_lr_scale``).
    """

    # Subclass MUST override with the full name→attr mapping.
    _NAME_TO_ATTR: dict

    def __init__(
        self,
        output_H: int,
        output_W: int,
        downsample_factor: int = 2,
        num_initial_gaussians: int = 100_000,
        object_recovery_initial: torch.Tensor | None = None,
        # Densification / pruning
        densify_grad_threshold: float = 5e-6,
        densify_scale_threshold: float = 0.01,
        densify_interval: int = 500,
        densify_until_step: int = 10_000,
        prune_weight_threshold: float = 0.005,
        opacity_aware_pruning: bool = True,
        max_gaussians: int = 500_000,
        render_batch_size: int = 4096,
        max_patch_radius: int = 8,
        init_scale: float = 5.0,
        min_scale: float = 0.5,
        # Common LR scales
        position_lr_scale: float = 1.0,
        weight_lr_scale: float = 1.0,
        label: str = "",
        **kwargs: Any,  # noqa: ARG002 — subclass-specific LR scales
    ) -> None:
        super().__init__()

        self.label = label
        self.output_H = output_H
        self.output_W = output_W
        self.downsample_factor = downsample_factor
        self.H_ds = output_H // downsample_factor
        self.W_ds = output_W // downsample_factor
        self.densify_grad_threshold = densify_grad_threshold
        self.densify_scale_threshold = densify_scale_threshold
        self.densify_interval = densify_interval
        self.densify_until_step = densify_until_step
        self.prune_weight_threshold = prune_weight_threshold
        self.opacity_aware_pruning = opacity_aware_pruning
        self.max_gaussians = max_gaussians
        self.render_batch_size = render_batch_size
        self.max_patch_radius = max_patch_radius
        self.init_scale = init_scale
        self.min_scale = max(0.05, min_scale)  # safety floor: det=σ⁴ underflows below 0.05
        self.position_lr_scale = position_lr_scale
        self.weight_lr_scale = weight_lr_scale

        # Pre-register local patch coordinate grid (constant buffer)
        local = torch.arange(
            -max_patch_radius, max_patch_radius + 1, dtype=torch.float32
        )
        local_y, local_x = torch.meshgrid(local, local, indexing="ij")
        self.register_buffer("_local_x", local_x.contiguous())
        self.register_buffer("_local_y", local_y.contiguous())
        self.register_buffer("_local_x_long", local_x.long().contiguous())
        self.register_buffer("_local_y_long", local_y.long().contiguous())

        # Smooth radial taper: cosine fade from 1→0 between 80%→100% of R.
        # Eliminates hard rectangular truncation artifacts for large Gaussians.
        R = float(max_patch_radius)
        r = torch.sqrt(local_x ** 2 + local_y ** 2)
        fade_start = 0.8
        t = torch.clamp((r / R - fade_start) / (1.0 - fade_start), 0.0, 1.0)
        taper = 0.5 * (1.0 + torch.cos(t * math.pi))
        self.register_buffer("_edge_taper", taper.contiguous())

        # Initialize Gaussian parameters (common + geometry)
        self._init_gaussians(num_initial_gaussians, object_recovery_initial)

        # Densification bookkeeping (non-parameter state)
        self._step_count = 0
        self._reset_densification_stats()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_gaussians(
        self,
        num_gaussians: int,
        initial_field: torch.Tensor | None,
    ) -> None:
        """Create common parameters and delegate geometry to subclass.

        Common parameters created here:
            * ``_xy`` — positions ``[N, 2]`` in pixel space
            * ``_weight`` — signed weights ``[N, 2]`` (real, imag)
            * ``_logit_opacity`` — opacity logits ``[N]``

        Then calls :meth:`_init_geometry_params` for subclass-specific
        geometry (e.g. scaling+rotation for RS, Cholesky factors for Cholesky).
        """
        H, W = self.output_H, self.output_W
        H_ds, W_ds = self.H_ds, self.W_ds

        # Random uniform positions in pixel space
        positions = torch.zeros(num_gaussians, 2)
        positions[:, 0] = torch.rand(num_gaussians) * W_ds
        positions[:, 1] = torch.rand(num_gaussians) * H_ds

        # Weights from initial field or constants
        if initial_field is not None and torch.is_complex(initial_field):
            col_idx = (positions[:, 0] / W_ds * (W - 1)).long().clamp(0, W - 1)
            row_idx = (positions[:, 1] / H_ds * (H - 1)).long().clamp(0, H - 1)
            sampled = initial_field[row_idx, col_idx]
            density = num_gaussians * (self.init_scale**2) / (H_ds * W_ds)
            overlap = max(density * math.pi, 1.0)
            w_real = sampled.real / overlap
            w_imag = sampled.imag / overlap
        else:
            w_real = torch.ones(num_gaussians) * 0.05
            w_imag = torch.randn(num_gaussians) * 0.001

        weights = torch.stack([w_real, w_imag], dim=-1)  # [N, 2]

        self._xy = nn.Parameter(positions)
        self._weight = nn.Parameter(weights)

        # Subclass creates geometry-specific parameters
        self._init_geometry_params(num_gaussians, self.init_scale)

        tag = f"[{self.label}] " if self.label else ""
        logger.info(
            "%sInitialized %d Gaussians (random, sigma=%.1f px) for %dx%d field (ds %dx%d)",
            tag,
            num_gaussians,
            self.init_scale,
            W,
            H,
            W_ds,
            H_ds,
        )

    def _reset_densification_stats(self) -> None:
        """Reset gradient accumulators for densification."""
        n = self.num_gaussians
        device = self._xy.device if self._xy.is_cuda else "cpu"
        self.xy_gradient_accum = torch.zeros(n, 1, device=device)
        self.denom = torch.zeros(n, 1, device=device)

    # ------------------------------------------------------------------
    # Abstract methods — subclass must implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _init_geometry_params(
        self, num_gaussians: int, init_scale: float
    ) -> None:
        """Create geometry-specific ``nn.Parameter`` attributes.

        Called at the end of :meth:`_init_gaussians`.  Must create the
        parameters that :attr:`_NAME_TO_ATTR` maps to (excluding the
        common ones: ``_xy``, ``_weight``, ``_logit_opacity``).
        """

    @abc.abstractmethod
    def _render_field(self) -> torch.Tensor:
        """Render Gaussians at downsampled resolution.

        Returns:
            Float tensor ``[2, H_ds, W_ds]`` — channel 0 real, channel 1
            imaginary.
        """

    @property
    @abc.abstractmethod
    def get_scaling(self) -> torch.Tensor:
        """Activated per-Gaussian scales ``[N, 2]`` for densification."""

    @abc.abstractmethod
    def _geometry_param_groups(self, base_lr: float):
        """Return optimizer param-group dicts for geometry parameters.

        Each dict must have ``"params"``, ``"lr"``, and ``"name"`` keys.
        The ``"name"`` values must match the geometry keys in
        :attr:`_NAME_TO_ATTR`.
        """

    @abc.abstractmethod
    def _create_split_children(
        self,
        selected: torch.Tensor,
        n_orig: int,
        N: int,
    ) -> dict:
        """Create parameters for split-densification children.

        Must return a dict with ALL keys from :attr:`_NAME_TO_ATTR`
        **except** ``"weight"`` and ``"opacity"`` (the base adds those).
        Must include ``"xy"`` (new child positions).

        Args:
            selected: Bool mask ``[n_orig]`` of parents to split.
            n_orig: Gaussian count before any densification this step.
            N: Number of children per parent.

        Returns:
            Dict mapping param-group names to new tensors.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_gaussians(self) -> int:
        """Current number of Gaussians."""
        return self._xy.shape[0]

    @property
    def get_xy(self) -> torch.Tensor:
        """Positions clamped to ``[0, W_ds-1] x [0, H_ds-1]``."""
        return torch.stack(
            [
                self._xy[:, 0].clamp(0.0, self.W_ds - 1),
                self._xy[:, 1].clamp(0.0, self.H_ds - 1),
            ],
            dim=-1,
        )

    @property
    def opacity(self) -> torch.Tensor:
        """Per-Gaussian opacity — always 1.0 (opacity feature removed)."""
        return torch.ones(self.num_gaussians, device=self._xy.device)

    fourier_upsample = staticmethod(_fourier_upsample)

    # ------------------------------------------------------------------
    # Shared rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_xy(xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Clamp XY positions to valid image bounds."""
        return torch.stack(
            [xy[:, 0].clamp(0.0, W - 1), xy[:, 1].clamp(0.0, H - 1)],
            dim=-1,
        )

    @staticmethod
    def _run_cuda_renderer(xy: torch.Tensor, render_fn):
        """Run CUDA renderer when available, otherwise return ``None``."""
        if not xy.is_cuda:
            return None
        try:
            from gscp.cuda import CUDA_AVAILABLE
        except ImportError:
            return None
        if not CUDA_AVAILABLE:
            return None
        return render_fn()

    def _prepare_patch_coordinates(
        self, xy_clamped: torch.Tensor, H: int, W: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare local patch indices used by PyTorch fallback renderers."""
        N = xy_clamped.shape[0]

        cx_f = xy_clamped[:, 0]
        cy_f = xy_clamped[:, 1]

        center_x = cx_f.detach().long()
        center_y = cy_f.detach().long()
        frac_x = cx_f - center_x.float()
        frac_y = cy_f - center_y.float()

        abs_x = center_x.view(N, 1, 1) + self._local_x_long.unsqueeze(0)
        abs_y = center_y.view(N, 1, 1) + self._local_y_long.unsqueeze(0)
        valid = (abs_x >= 0) & (abs_x < W) & (abs_y >= 0) & (abs_y < H)
        flat_idx = (abs_y * W + abs_x).clamp(0, H * W - 1).reshape(-1)
        return frac_x, frac_y, valid, flat_idx

    @staticmethod
    def _scatter_complex_channels(
        weight: torch.Tensor,
        opacity: torch.Tensor,
        gauss_masked: torch.Tensor,
        flat_idx: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Scatter-add weighted Gaussian patches to output image channels."""
        device = gauss_masked.device
        N = weight.shape[0]

        w_real = (weight[:, 0] * opacity).view(N, 1, 1) * gauss_masked
        real_img = (
            torch.zeros(H * W, device=device, dtype=w_real.dtype)
            .scatter_add(0, flat_idx, w_real.reshape(-1))
            .view(H, W)
        )

        w_imag = (weight[:, 1] * opacity).view(N, 1, 1) * gauss_masked
        imag_img = (
            torch.zeros(H * W, device=device, dtype=w_imag.dtype)
            .scatter_add(0, flat_idx, w_imag.reshape(-1))
            .view(H, W)
        )
        return torch.stack([real_img, imag_img], dim=0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self) -> torch.Tensor:
        """Render Gaussian field and return one complex field tensor.

        The returned tensor is geometry-agnostic: the trainer decides whether
        this field is treated as object / probe / CS.
        """
        rendered = self._render_field()  # [2, H_ds, W_ds]
        complex_ds = torch.complex(rendered[0], rendered[1])

        if self.downsample_factor > 1:
            object_recovery = (
                self.fourier_upsample(
                    complex_ds.unsqueeze(0).unsqueeze(0),
                    self.downsample_factor,
                )
                .squeeze(0)
                .squeeze(0)
            )
        else:
            object_recovery = complex_ds

        self._step_count += 1
        return object_recovery

    # ------------------------------------------------------------------
    # Per-param-group optimizer support
    # ------------------------------------------------------------------

    def get_param_groups(self, base_lr: float):
        """Return per-parameter-group dicts for the optimizer.

        Combines common groups (xy, weight, opacity) with geometry-specific
        groups from :meth:`_geometry_param_groups`.
        """
        groups = [
            {
                "params": [self._xy],
                "lr": base_lr * self.position_lr_scale,
                "name": "xy",
            },
            {
                "params": [self._weight],
                "lr": base_lr * self.weight_lr_scale,
                "name": "weight",
            },
        ]
        groups.extend(self._geometry_param_groups(base_lr))

        # C1: Validate _NAME_TO_ATTR consistency
        group_names = {g["name"] for g in groups}
        expected = set(self._NAME_TO_ATTR.keys())
        if group_names != expected:
            raise RuntimeError(
                f"_NAME_TO_ATTR keys {expected} != param group names "
                f"{group_names}. Subclass {type(self).__name__} must keep "
                f"_NAME_TO_ATTR in sync with get_param_groups()."
            )

        return groups

    # ------------------------------------------------------------------
    # Densification and pruning
    # ------------------------------------------------------------------

    def accumulate_gradients(self) -> None:
        """Accumulate position-gradient norms for densification.

        Must be called after ``loss.backward()`` and before
        ``clip_grad_norm_`` or ``optimizer.step()``.
        """
        if self._xy.grad is not None:
            grad_norm = torch.norm(self._xy.grad, dim=-1, keepdim=True)
            if self.xy_gradient_accum.device != grad_norm.device:
                self.xy_gradient_accum = self.xy_gradient_accum.to(
                    grad_norm.device
                )
                self.denom = self.denom.to(grad_norm.device)
            self.xy_gradient_accum[: grad_norm.shape[0]] += grad_norm
            self.denom[: grad_norm.shape[0]] += 1

    def densification_step(self, optimizer) -> None:
        """Conditionally densify and prune Gaussians."""
        if self._step_count % self.densify_interval != 0:
            return
        if self._step_count > self.densify_until_step:
            return
        if self.num_gaussians >= self.max_gaussians:
            return

        grads = self.xy_gradient_accum / (self.denom + 1e-8)
        grads[grads.isnan()] = 0.0
        n_orig = grads.shape[0]

        self._densify_and_clone(grads, n_orig, optimizer)
        self._densify_and_split(grads, n_orig, optimizer)
        self._prune(optimizer)

        # Safety net: prune lowest-contribution if still over cap
        if self.num_gaussians > self.max_gaussians:
            excess = self.num_gaussians - self.max_gaussians
            weight_mag = self._weight.detach().abs().sum(dim=-1)
            _, to_prune = weight_mag.topk(excess, largest=False)
            mask = torch.zeros(
                self.num_gaussians, dtype=torch.bool, device=self._xy.device
            )
            mask[to_prune] = True
            self._prune_by_mask(mask, optimizer)
            logger.warning(
                "%sPost-densification cap enforced: pruned %d excess Gaussians",
                f"[{self.label}] " if self.label else "",
                excess,
            )

        self._reset_densification_stats()
        tag = f"[{self.label}] " if self.label else ""
        logger.info(
            "%sDensification at step %d: %d Gaussians",
            tag,
            self._step_count,
            self.num_gaussians,
        )

    def _apply_budget(self, selected: torch.Tensor, grads: torch.Tensor, budget: int) -> bool:
        """Trim *selected* in-place to at most *budget* entries (top-gradient).

        Returns ``False`` if nothing is selected or budget is exhausted.
        """
        if not selected.any() or budget <= 0:
            return False
        if selected.sum() > budget:
            indices = torch.where(selected)[0]
            grad_vals = grads.squeeze()[indices]
            _, top_k = grad_vals.topk(budget)
            selected.zero_()
            selected[indices[top_k]] = True
        return True

    def _densify_and_clone(self, grads: torch.Tensor, n_orig: int, optimizer) -> None:
        """Clone small, under-reconstructed Gaussians."""
        max_scale = self.get_scaling[:n_orig].max(dim=1).values
        selected = (grads.squeeze() >= self.densify_grad_threshold) & (
            max_scale
            <= self.densify_scale_threshold * max(self.output_H, self.output_W)
        )

        budget = self.max_gaussians - self.num_gaussians
        if not self._apply_budget(selected, grads, budget):
            return

        # Clone all params via _NAME_TO_ATTR
        new_params = {}
        for name, attr in self._NAME_TO_ATTR.items():
            new_params[name] = getattr(self, attr)[selected].detach().clone()

        # Halve weight for parent and child
        new_params["weight"] = new_params["weight"] * 0.5
        self._weight.data[selected] *= 0.5

        self._append_gaussians(optimizer, new_params)

    def _densify_and_split(
        self,
        grads: torch.Tensor,
        n_orig: int,
        optimizer,
        N: int = 2,
    ) -> None:
        """Split large, under-reconstructed Gaussians into *N* children."""
        max_scale = self.get_scaling[:n_orig].max(dim=1).values
        selected = (grads.squeeze() >= self.densify_grad_threshold) & (
            max_scale
            > self.densify_scale_threshold * max(self.output_H, self.output_W)
        )

        budget = (self.max_gaussians - self.num_gaussians) // (N - 1)
        if not self._apply_budget(selected, grads, budget):
            return

        # Subclass creates geometry-specific children (xy + geometry params)
        new_params = self._create_split_children(selected, n_orig, N)

        # Base adds common weight
        new_params["weight"] = (
            self._weight[:n_orig][selected].detach().repeat(N, 1) / N
        )

        self._append_gaussians(optimizer, new_params)

        # Prune the original (now-split) Gaussians
        prune_mask = torch.zeros(
            self.num_gaussians, dtype=torch.bool, device=self._xy.device
        )
        orig_indices = torch.where(selected)[0]
        prune_mask[orig_indices] = True
        self._prune_by_mask(prune_mask, optimizer)

    def _prune(self, optimizer) -> None:
        """Prune Gaussians with negligible weight magnitude."""
        weight_mag = self._weight.detach().abs().sum(dim=-1)
        prune_mask = weight_mag < self.prune_weight_threshold
        if prune_mask.any():
            self._prune_by_mask(prune_mask, optimizer)

    # ------------------------------------------------------------------
    # Optimizer-aware Gaussian manipulation
    # ------------------------------------------------------------------

    def _param_map(self):
        """Map ``data_ptr → internal_name`` for this model's parameters.

        Matches optimizer groups by **parameter identity** rather than
        group name, so models sharing an optimizer with prefixed group
        names (e.g. ``"cs_xy"`` vs ``"xy"``) never collide.
        """
        result = {}
        for name, attr in self._NAME_TO_ATTR.items():
            param = getattr(self, attr, None)
            if param is not None:
                result[param.data_ptr()] = name
        return result

    def _append_gaussians(
        self,
        optimizer,
        new_params: dict,
    ) -> None:
        """Append new Gaussians to all parameters and extend optimizer state.

        Groups are identified by **parameter identity** (data_ptr),
        not group name, so shared optimizers with prefixed group names
        are safe.

        Args:
            optimizer: Adam optimizer whose state is extended in-place.
            new_params: Dict mapping internal names to new tensors.
                Must contain all keys from :attr:`_NAME_TO_ATTR`.
        """
        pmap = self._param_map()

        for group in optimizer.param_groups:
            ptr = group["params"][0].data_ptr()
            internal_name = pmap.get(ptr)
            if internal_name is None or internal_name not in new_params:
                continue

            ext = new_params[internal_name]
            stored = optimizer.state.get(group["params"][0], None)

            if stored is not None:
                stored["exp_avg"] = torch.cat(
                    [stored["exp_avg"], torch.zeros_like(ext)], dim=0
                )
                stored["exp_avg_sq"] = torch.cat(
                    [stored["exp_avg_sq"], torch.zeros_like(ext)], dim=0
                )
                del optimizer.state[group["params"][0]]

            old_param = group["params"][0]
            new_param = nn.Parameter(
                torch.cat([old_param.detach(), ext], dim=0).requires_grad_(
                    True
                )
            )
            group["params"][0] = new_param

            if stored is not None:
                optimizer.state[new_param] = stored

            # Rebind attribute immediately (data_ptr changed)
            attr = self._NAME_TO_ATTR[internal_name]
            setattr(self, attr, new_param)

        # Extend gradient accumulators
        n_new = new_params["xy"].shape[0]
        device = self._xy.device
        self.xy_gradient_accum = torch.cat(
            [self.xy_gradient_accum, torch.zeros(n_new, 1, device=device)],
            dim=0,
        )
        self.denom = torch.cat(
            [self.denom, torch.zeros(n_new, 1, device=device)], dim=0
        )

    def _prune_by_mask(
        self,
        mask: torch.Tensor,
        optimizer,
    ) -> None:
        """Remove Gaussians where ``mask`` is ``True``.

        Groups are identified by **parameter identity** (data_ptr),
        not group name, so shared optimizers are safe.
        """
        keep = ~mask
        pmap = self._param_map()

        for group in optimizer.param_groups:
            ptr = group["params"][0].data_ptr()
            internal_name = pmap.get(ptr)
            if internal_name is None:
                continue

            stored = optimizer.state.get(group["params"][0], None)
            if stored is not None:
                stored["exp_avg"] = stored["exp_avg"][keep]
                stored["exp_avg_sq"] = stored["exp_avg_sq"][keep]
                del optimizer.state[group["params"][0]]

            new_param = nn.Parameter(
                group["params"][0].detach()[keep].requires_grad_(True)
            )
            group["params"][0] = new_param

            if stored is not None:
                optimizer.state[new_param] = stored

            # Rebind attribute immediately (data_ptr changed)
            attr = self._NAME_TO_ATTR[internal_name]
            setattr(self, attr, new_param)

        self.xy_gradient_accum = self.xy_gradient_accum[keep]
        self.denom = self.denom[keep]

    # ------------------------------------------------------------------
    # MCMC density control (3DGS-MCMC style, paper 2404.09591)
    # ------------------------------------------------------------------

    def mcmc_relocation_step(self, optimizer) -> None:
        """MCMC-style density control: grow then relocate dead Gaussians.

        Replaces heuristic ADC (clone/split/prune) with:
        1. Growth: birth new Gaussians from high-weight sources (if below cap).
        2. Relocation: teleport lowest-weight Gaussians to high-weight sources.

        Requires ``mcmc_grow_rate``, ``mcmc_relocation_fraction``, and
        ``densify_interval`` / ``densify_until_step`` attributes on ``self``.
        """
        if self._step_count % self.densify_interval != 0:
            return
        if self._step_count > self.densify_until_step:
            return

        N = self.num_gaussians
        w_mag = self._weight.detach().norm(dim=-1)

        # 1. Growth: add Gaussians if below cap
        n_to_add = min(
            int(self.mcmc_grow_rate * N),
            self.max_gaussians - N,
        )
        if n_to_add > 0:
            self._birth_gaussians(optimizer, n_to_add, w_mag)
            # Refresh after birth
            N = self.num_gaussians
            w_mag = self._weight.detach().norm(dim=-1)

        # 2. Relocate dead → alive
        # Stuck Gaussians (clamped at floor or ceiling) are treated as dead
        # regardless of weight, preventing population traps at clamp boundaries.
        relocation_score = w_mag.clone()
        with torch.no_grad():
            max_scale = self.get_scaling.max(dim=1).values
            max_sigma = float(self.max_patch_radius) / 1.5
            stuck_floor = max_scale <= self.min_scale * 1.05
            stuck_ceil = max_scale >= max_sigma * 0.95
            stuck = stuck_floor | stuck_ceil
            relocation_score[stuck] = 0.0
        n_relocate = int(self.mcmc_relocation_fraction * N)
        n_stuck = stuck.sum().item()
        if n_relocate > 0:
            self._relocate_dead(optimizer, n_relocate, relocation_score)

        self._reset_densification_stats()

        tag = f"[{self.label}] " if self.label else ""
        logger.info(
            "%sMCMC step=%d | N=%d (grew +%d, relocated %d, stuck_scale=%d)",
            tag, self._step_count, self.num_gaussians,
            n_to_add, n_relocate, n_stuck,
        )

    def _birth_gaussians(
        self, optimizer, n_to_add: int, w_mag: torch.Tensor,
    ) -> None:
        """Sample alive Gaussians, duplicate with noise, append.

        Weight conservation: when a source is sampled k times, both the
        parent and each child receive ``w / (k+1)``.
        """
        probs = w_mag / (w_mag.sum() + 1e-8)
        source_idx = torch.multinomial(probs, n_to_add, replacement=True)

        # Count duplicates per source for correct weight splitting
        N = self.num_gaussians
        device = self._xy.device
        counts = torch.zeros(N, device=device)
        counts.scatter_add_(
            0, source_idx,
            torch.ones(n_to_add, device=device),
        )

        scales = self.get_scaling[source_idx]
        noise = torch.randn(n_to_add, 2, device=device) * scales

        # Each child gets w_parent / (k+1)
        child_factor = 1.0 / (counts[source_idx] + 1.0)

        # Build new_params from all _NAME_TO_ATTR entries
        # Skip 'opacity' — MCMC doesn't use it (zhongtian alignment)
        new_params: dict[str, torch.Tensor] = {}
        for name, attr in self._NAME_TO_ATTR.items():
            if name == "opacity":
                continue
            param = getattr(self, attr)
            new_params[name] = param.data[source_idx].clone()

        # Position: add noise
        new_params["xy"] = new_params["xy"] + noise

        # Scale perturbation: break birth echo (log-space noise ≈ ±10%)
        for scale_name in ("scaling", "log_L_diag"):
            if scale_name in new_params:
                scale_noise = torch.randn_like(new_params[scale_name]) * 0.1
                new_params[scale_name] = new_params[scale_name] + scale_noise

        # Weight: child gets w / (k+1)
        new_params["weight"] = (
            new_params["weight"] * child_factor.unsqueeze(-1)
        )

        # Parent: w *= 1/(k+1)
        sampled_mask = counts > 0
        parent_factor = 1.0 / (counts + 1.0)
        self._weight.data[sampled_mask] *= (
            parent_factor[sampled_mask].unsqueeze(-1)
        )

        self._append_gaussians(optimizer, new_params)

    def _relocate_dead(
        self, optimizer, n_relocate: int, w_mag: torch.Tensor,
    ) -> None:
        """In-place relocation of lowest-weight Gaussians to alive sources."""
        N = self.num_gaussians
        n_relocate = min(n_relocate, N - 1)  # keep at least 1 alive
        if n_relocate <= 0:
            return

        _, dead_indices = w_mag.topk(n_relocate, largest=False)

        alive_mask = torch.ones(N, dtype=torch.bool, device=self._xy.device)
        alive_mask[dead_indices] = False
        alive_indices = torch.where(alive_mask)[0]

        if len(alive_indices) == 0:
            return

        alive_w_mag = w_mag[alive_indices]
        probs = alive_w_mag / (alive_w_mag.sum() + 1e-8)
        sampled = torch.multinomial(probs, n_relocate, replacement=True)
        source_alive = alive_indices[sampled]

        # Weight splitting: count duplicates per alive source
        counts = torch.zeros(N, device=self._xy.device)
        counts.scatter_add_(
            0, source_alive,
            torch.ones(n_relocate, device=self._xy.device),
        )

        with torch.no_grad():
            # Relocate positions with noise
            scales = self.get_scaling[source_alive]
            noise = torch.randn(n_relocate, 2, device=self._xy.device) * scales
            self._xy.data[dead_indices] = self._xy.data[source_alive] + noise

            # Copy all geometry params from source to dead
            # Skip 'opacity' — MCMC doesn't use it (zhongtian alignment)
            for name, attr in self._NAME_TO_ATTR.items():
                if name in ("xy", "weight", "opacity"):
                    continue  # handled separately or skipped
                param = getattr(self, attr)
                param.data[dead_indices] = param.data[source_alive].clone()

            # Scale perturbation: break relocation echo (log-space ±10%)
            # Without this, relocated Gaussians inherit source's exact scale
            # and may immediately get stuck at clamp boundaries again.
            for scale_name in ("scaling", "log_L_diag"):
                attr = self._NAME_TO_ATTR.get(scale_name)
                if attr is not None and hasattr(self, attr):
                    param = getattr(self, attr)
                    scale_noise = torch.randn(n_relocate, param.shape[1], device=param.device) * 0.1
                    param.data[dead_indices] += scale_noise

            # Weight splitting with duplicate handling
            scale_factor = 1.0 / (counts + 1.0)
            sampled_mask = counts > 0
            dead_weight = (
                self._weight.data[source_alive]
                * scale_factor[source_alive].unsqueeze(-1)
            )
            self._weight.data[sampled_mask] *= (
                scale_factor[sampled_mask].unsqueeze(-1)
            )
            self._weight.data[dead_indices] = dead_weight

        # Reset Adam state for relocated Gaussians
        self._reset_adam_state_indices(optimizer, dead_indices)

    def _reset_adam_state_indices(self, optimizer, indices: torch.Tensor) -> None:
        """Zero out Adam moments for specific Gaussian indices."""
        pmap = self._param_map()
        for group in optimizer.param_groups:
            ptr = group["params"][0].data_ptr()
            if pmap.get(ptr) is None:
                continue
            stored = optimizer.state.get(group["params"][0], None)
            if stored is not None:
                stored["exp_avg"][indices] = 0
                stored["exp_avg_sq"][indices] = 0


# Backward-compatible alias (old name).
BaseGaussianField = GaussianFieldCore
