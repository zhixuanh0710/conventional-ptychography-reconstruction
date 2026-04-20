"""Unified 2D Gaussian Splatting model.

This implementation supports two geometry parameterizations:

- ``parameterization="rs"``: rotation + scaling
- ``parameterization="cholesky"``: Cholesky factor of covariance

Weight representation is also configurable:

- ``weight_representation="real_imag"``: Cartesian weights (default)
- ``weight_representation="amplitude_phase"``: Polar weights

All modes share the same optimizer/densification skeleton from
``GaussianFieldCore`` and differ only in geometry-specific math.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from gscp.models.base_gaussian_field import GaussianFieldCore

logger = logging.getLogger(__name__)

# Default minimum activated scale (sigma) in pixels.
# Now a runtime parameter — passed to CUDA as min_scale, stored as self.min_scale.
_DEFAULT_MIN_SCALE: float = 0.5


class GaussianFieldModel(GaussianFieldCore):
    """Gaussian splatting model with selectable geometry parameterization."""

    _RS_NAME_TO_ATTR = {
        "xy": "_xy",
        "scaling": "_scaling",
        "rotation": "_rotation",
        "weight": "_weight",
    }
    _CHOLESKY_NAME_TO_ATTR = {
        "xy": "_xy",
        "log_L_diag": "_log_L_diag",
        "L_offdiag": "_L_offdiag",
        "weight": "_weight",
    }
    _NAME_TO_ATTR = _RS_NAME_TO_ATTR

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
        min_scale: float = _DEFAULT_MIN_SCALE,
        # Per-parameter-group learning rate scales
        position_lr_scale: float = 1.0,
        scaling_lr_scale: float = 1.0,
        rotation_lr_scale: float = 1.0,
        cholesky_lr_scale: float = 1.0,
        weight_lr_scale: float = 1.0,
        parameterization: str = "rs",
        weight_representation: str = "real_imag",
        phase_init_std: float = 0.3,
        # MCMC density control (replaces ADC when enabled)
        density_control: str = "adc",
        mcmc_grow_rate: float = 0.05,
        mcmc_relocation_fraction: float = 0.05,
        mcmc_noise_lr_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        parameterization = parameterization.lower()
        if parameterization not in ("rs", "cholesky"):
            raise ValueError(
                f"Unknown parameterization '{parameterization}'. "
                "Expected 'rs' or 'cholesky'."
            )
        self.parameterization = parameterization
        self._NAME_TO_ATTR = (
            self._CHOLESKY_NAME_TO_ATTR
            if self.parameterization == "cholesky"
            else self._RS_NAME_TO_ATTR
        )
        self.scaling_lr_scale = scaling_lr_scale
        self.rotation_lr_scale = rotation_lr_scale
        self.cholesky_lr_scale = cholesky_lr_scale
        self.phase_init_std = float(phase_init_std)
        self.density_control = str(density_control).lower()
        self.mcmc_grow_rate = float(mcmc_grow_rate)
        self.mcmc_relocation_fraction = float(mcmc_relocation_fraction)
        self.mcmc_noise_lr_scale = float(mcmc_noise_lr_scale)

        wr = weight_representation.lower()
        alias = {
            "cartesian": "real_imag",
            "real_imag": "real_imag",
            "polar": "amplitude_phase",
            "amplitude_phase": "amplitude_phase",
        }
        if wr not in alias:
            raise ValueError(
                f"Unknown weight_representation '{weight_representation}'. "
                "Expected 'real_imag' or 'amplitude_phase'."
            )
        self.weight_representation = alias[wr]

        super().__init__(
            output_H=output_H,
            output_W=output_W,
            downsample_factor=downsample_factor,
            num_initial_gaussians=num_initial_gaussians,
            object_recovery_initial=object_recovery_initial,
            densify_grad_threshold=densify_grad_threshold,
            densify_scale_threshold=densify_scale_threshold,
            densify_interval=densify_interval,
            densify_until_step=densify_until_step,
            prune_weight_threshold=prune_weight_threshold,
            opacity_aware_pruning=opacity_aware_pruning,
            max_gaussians=max_gaussians,
            render_batch_size=render_batch_size,
            max_patch_radius=max_patch_radius,
            init_scale=init_scale,
            min_scale=min_scale,
            position_lr_scale=position_lr_scale,
            weight_lr_scale=weight_lr_scale,
            **kwargs,
        )

    def _init_gaussians(
        self,
        num_gaussians: int,
        initial_field: torch.Tensor | None,
    ) -> None:
        """Initialize base Gaussian params, optionally converting to polar weights."""
        super()._init_gaussians(num_gaussians, initial_field)

        if self.weight_representation != "amplitude_phase":
            return

        w_real = self._weight.data[:, 0]
        w_imag = self._weight.data[:, 1]
        if initial_field is not None and torch.is_complex(initial_field):
            amplitude = torch.sqrt(w_real ** 2 + w_imag ** 2)
            phase = torch.atan2(w_imag, w_real)
        else:
            amplitude = w_real.abs()
            phase = torch.zeros_like(amplitude)

        phase = phase + torch.randn_like(phase) * self.phase_init_std
        self._weight.data[:, 0] = amplitude
        self._weight.data[:, 1] = phase

        logger.info(
            "Polar weight init: A_mean=%.4f, phi_std=%.4f",
            amplitude.mean().item(),
            phase.std().item(),
        )

    def _weights_as_cartesian(self, weight: torch.Tensor) -> torch.Tensor:
        """Return rendering weights for the two-channel splatting kernel.

        For ``real_imag``: weights are already (real, imag) — pass through.
        For ``amplitude_phase``: weights are (amplitude, phase) — pass
        through as-is.  The two channels are splatted independently and
        combined as ``amp * exp(i * phase)`` in :meth:`forward`.
        """
        return weight

    def _init_geometry_params(
        self, num_gaussians: int, init_scale: float
    ) -> None:
        """Create geometry parameters based on selected parameterization."""
        if self.parameterization == "cholesky":
            self._init_geometry_cholesky(num_gaussians, init_scale)
            return
        self._init_geometry_rs(num_gaussians, init_scale)

    def _init_geometry_rs(self, num_gaussians: int, init_scale: float) -> None:
        # Inverse sigmoid activation:
        # sigma = min + (max - min) * sigmoid(raw)  =>  raw = logit((sigma - min)/(max - min))
        max_sigma = float(self.max_patch_radius)
        init_clamped = max(
            self.min_scale + 1e-4,
            min(max_sigma - 1e-4, float(init_scale)),
        )
        t = (init_clamped - self.min_scale) / (max_sigma - self.min_scale)
        init_raw = math.log(t / (1.0 - t))
        scales = (
            torch.full((num_gaussians, 2), init_raw)
            + torch.randn(num_gaussians, 2) * 0.1
        )
        rotations = torch.randn(num_gaussians, 1) * 0.1

        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rotations)

    def _init_geometry_cholesky(self, num_gaussians: int, init_scale: float) -> None:
        init_log = math.log(init_scale)
        log_L_diag = (
            torch.full((num_gaussians, 2), init_log)
            + torch.randn(num_gaussians, 2) * 0.1
        )
        L_offdiag = torch.zeros(num_gaussians)
        self._log_L_diag = nn.Parameter(log_L_diag)
        self._L_offdiag = nn.Parameter(L_offdiag)

    @property
    def L_diag(self) -> torch.Tensor:
        """Positive Cholesky diagonal ``[N, 2]``."""
        if self.parameterization != "cholesky":
            raise AttributeError("L_diag is only available in cholesky mode")
        return torch.exp(self._log_L_diag)

    @property
    def get_scaling(self) -> torch.Tensor:
        """Effective per-axis scales used by densification.

        RS mode: sigmoid activation — sigma = min + (max - min) * sigmoid(raw).
        """
        max_sigma = float(self.max_patch_radius)
        if self.parameterization == "rs":
            return self.min_scale + (max_sigma - self.min_scale) * torch.sigmoid(
                self._scaling
            )
        # Cholesky fallback (unused but kept for compat)
        l_diag = torch.clamp(
            torch.exp(self._log_L_diag),
            min=self.min_scale,
            max=max_sigma,
        )
        l21 = self._L_offdiag
        sigma_x = l_diag[:, 0]
        sigma_y = torch.sqrt(l21 ** 2 + l_diag[:, 1] ** 2)
        return torch.stack([sigma_x, sigma_y], dim=-1)

    def _render_field(self) -> torch.Tensor:
        """Render Gaussians at downsampled resolution."""
        if self.parameterization == "cholesky":
            return self._render_gaussians_cholesky(
                self._xy,
                self._log_L_diag,
                self._L_offdiag,
                self._weight,
                self.H_ds,
                self.W_ds,
            )
        return self._render_gaussians(
            self._xy,
            self._scaling,
            self._rotation,
            self._weight,
            self.H_ds,
            self.W_ds,
        )

    def forward(self) -> torch.Tensor:
        """Render Gaussian field and return one complex field tensor.

        For ``real_imag`` mode the two rendered channels are (real, imag)
        and combined as ``real + i*imag``.

        For ``amplitude_phase`` mode the two rendered channels are
        (amplitude, phase) and combined as ``amp * exp(i * phase)``.
        """
        rendered = self._render_field()  # [2, H_ds, W_ds]

        if self.weight_representation == "amplitude_phase":
            complex_ds = rendered[0] * torch.exp(1j * rendered[1])
        else:
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

    def _render_gaussians(
        self,
        xy: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        weight: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Render RS Gaussians to ``[2, H, W]`` (no opacity).

        Sigmoid activation: σ_activated = min + (max_patch − min) · sigmoid(raw).
        Output strictly in (min, max_patch), so the CUDA kernel's clamp is
        always a no-op and gradients are never zeroed at the boundary.
        The kernel expects log-space scaling, so ``log(σ_activated)`` is
        passed to it — inside the kernel ``exp(log(σ)) = σ``.
        """
        xy_clamped = self._clamp_xy(xy, H, W)
        weight_cart = self._weights_as_cartesian(weight)

        max_patch = float(self.max_patch_radius)
        sigma = self.min_scale + (max_patch - self.min_scale) * torch.sigmoid(scaling)
        scaling_log = torch.log(sigma)

        def _cuda_call():
            from gscp.cuda._wrapper import GaussianRasterize2D

            return GaussianRasterize2D.apply(
                xy_clamped,
                scaling_log,
                rotation,
                weight_cart,
                H,
                W,
                self.max_patch_radius,
                self.min_scale,
            )

        cuda_out = self._run_cuda_renderer(xy, _cuda_call)
        if cuda_out is not None:
            return cuda_out

        # Pure-PyTorch fallback uses sigma directly (no exp needed).
        opacity_ones = torch.ones(xy_clamped.shape[0], device=xy_clamped.device)
        return self._render_gaussians_pytorch(
            xy_clamped,
            sigma,
            rotation,
            weight_cart,
            opacity_ones,
            H,
            W,
        )

    def _render_gaussians_cholesky(
        self,
        xy: torch.Tensor,
        log_L_diag: torch.Tensor,
        L_offdiag: torch.Tensor,
        weight: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Render Cholesky Gaussians to ``[2, H, W]`` (no opacity)."""
        xy_clamped = self._clamp_xy(xy, H, W)
        weight_cart = self._weights_as_cartesian(weight)

        def _cuda_call():
            from gscp.cuda._wrapper import GaussianRasterize2DCholesky

            return GaussianRasterize2DCholesky.apply(
                xy_clamped,
                log_L_diag,
                L_offdiag,
                weight_cart,
                H,
                W,
                self.max_patch_radius,
                self.min_scale,
            )

        cuda_out = self._run_cuda_renderer(xy, _cuda_call)
        if cuda_out is not None:
            return cuda_out

        pytorch_max_scale = float(self.max_patch_radius) / 1.5
        l_diag = torch.clamp(
            torch.exp(log_L_diag),
            min=self.min_scale,
            max=pytorch_max_scale,
        )
        opacity_ones = torch.ones(xy_clamped.shape[0], device=xy_clamped.device)
        return self._render_gaussians_pytorch_cholesky(
            xy_clamped,
            l_diag,
            L_offdiag,
            weight_cart,
            opacity_ones,
            H,
            W,
        )

    def _render_field_pytorch(self) -> torch.Tensor:
        """Render using the pure-PyTorch path (for parity tests)."""
        if self.parameterization == "cholesky":
            pytorch_max_scale = float(self.max_patch_radius) / 1.5
            l_diag = torch.clamp(
                torch.exp(self._log_L_diag),
                min=self.min_scale,
                max=pytorch_max_scale,
            )
            return self._render_gaussians_pytorch_cholesky(
                self.get_xy,
                l_diag,
                self._L_offdiag,
                self._weights_as_cartesian(self._weight),
                self.opacity,
                self.H_ds,
                self.W_ds,
            )

        pytorch_max_scale = float(self.max_patch_radius) / 1.5
        activated_scaling = torch.clamp(
            torch.exp(self._scaling),
            min=self.min_scale,
            max=pytorch_max_scale,
        )
        return self._render_gaussians_pytorch(
            self.get_xy,
            activated_scaling,
            self._rotation,
            self._weights_as_cartesian(self._weight),
            self.opacity,
            self.H_ds,
            self.W_ds,
        )

    def _render_gaussians_pytorch(
        self,
        xy_clamped: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        weight: torch.Tensor,
        opacity: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Pure-PyTorch RS renderer using vectorised scatter-add."""
        N = xy_clamped.shape[0]
        frac_x, frac_y, valid, flat_idx = self._prepare_patch_coordinates(
            xy_clamped, H, W
        )

        cos_a = torch.cos(rotation[:, 0]).view(N, 1, 1)
        sin_a = torch.sin(rotation[:, 0]).view(N, 1, 1)
        sx = scaling[:, 0].view(N, 1, 1)
        sy = scaling[:, 1].view(N, 1, 1)

        dx = self._local_x.unsqueeze(0) - frac_x.view(N, 1, 1)
        dy = self._local_y.unsqueeze(0) - frac_y.view(N, 1, 1)

        dx_rot = cos_a * dx + sin_a * dy
        dy_rot = -sin_a * dx + cos_a * dy

        gauss = torch.exp(
            -0.5 * (dx_rot**2 / (sx**2 + 1e-8) + dy_rot**2 / (sy**2 + 1e-8))
        )

        gauss_masked = gauss * valid.float() * self._edge_taper.unsqueeze(0)
        return self._scatter_complex_channels(
            weight=weight,
            opacity=opacity,
            gauss_masked=gauss_masked,
            flat_idx=flat_idx,
            H=H,
            W=W,
        )

    def _render_gaussians_pytorch_cholesky(
        self,
        xy_clamped: torch.Tensor,
        l_diag: torch.Tensor,
        L_offdiag: torch.Tensor,
        weight: torch.Tensor,
        opacity: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Pure-PyTorch Cholesky renderer using forward substitution."""
        N = xy_clamped.shape[0]
        frac_x, frac_y, valid, flat_idx = self._prepare_patch_coordinates(
            xy_clamped, H, W
        )

        l11 = l_diag[:, 0].view(N, 1, 1)
        l22 = l_diag[:, 1].view(N, 1, 1)
        l21 = L_offdiag.view(N, 1, 1)

        dx = self._local_x.unsqueeze(0) - frac_x.view(N, 1, 1)
        dy = self._local_y.unsqueeze(0) - frac_y.view(N, 1, 1)

        z1 = dx / (l11 + 1e-8)
        z2 = (dy - l21 * z1) / (l22 + 1e-8)
        gauss = torch.exp(-0.5 * (z1 ** 2 + z2 ** 2))

        gauss_masked = gauss * valid.float() * self._edge_taper.unsqueeze(0)
        return self._scatter_complex_channels(
            weight=weight,
            opacity=opacity,
            gauss_masked=gauss_masked,
            flat_idx=flat_idx,
            H=H,
            W=W,
        )

    def _geometry_param_groups(self, base_lr: float) -> list[dict[str, Any]]:
        """Return geometry parameter groups for selected parameterization."""
        if self.parameterization == "cholesky":
            return [
                {
                    "params": [self._log_L_diag],
                    "lr": base_lr * self.cholesky_lr_scale,
                    "name": "log_L_diag",
                },
                {
                    "params": [self._L_offdiag],
                    "lr": base_lr * self.cholesky_lr_scale,
                    "name": "L_offdiag",
                },
            ]
        return [
            {
                "params": [self._scaling],
                "lr": base_lr * self.scaling_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": base_lr * self.rotation_lr_scale,
                "name": "rotation",
            },
        ]

    # ------------------------------------------------------------------
    # Densification: ADC vs MCMC
    # ------------------------------------------------------------------

    def densification_step(self, optimizer) -> None:
        """ADC (default) or MCMC density control."""
        if self.density_control == "mcmc":
            self.mcmc_relocation_step(optimizer)
        else:
            super().densification_step(optimizer)

    def sgld_noise_step(self, lr_xy: float, lr_scaling: float) -> None:
        """Inject SGLD noise on positions and scales (MCMC mode only).

        Noise is gated by ``densify_until_step`` so the model can converge
        cleanly during the refinement phase.
        """
        if self.density_control != "mcmc":
            return
        if self._step_count > self.densify_until_step:
            return  # No noise during refinement phase

        noise_lr_scale = self.mcmc_noise_lr_scale

        with torch.no_grad():
            w_mag = self._weight.detach().norm(dim=-1)
            noise_scale = torch.sigmoid(1.0 - w_mag / (w_mag.max() + 1e-8))

            # Position noise
            pos_noise = (
                torch.randn_like(self._xy.data)
                * math.sqrt(2 * lr_xy)
                * noise_lr_scale
            )
            self._xy.data.add_(pos_noise * noise_scale.unsqueeze(-1))
            self._xy.data[:, 0].clamp_(0, self.W_ds - 1)
            self._xy.data[:, 1].clamp_(0, self.H_ds - 1)

            # Scale noise with reflecting boundaries
            if self.parameterization == "cholesky":
                scale_param = self._log_L_diag
            else:
                scale_param = self._scaling

            s_noise = (
                torch.randn_like(scale_param.data)
                * math.sqrt(2 * lr_scaling)
                * noise_lr_scale
            )
            scale_param.data.add_(s_noise * noise_scale.unsqueeze(-1))

            # Reflecting boundaries: bounce off log(min_scale) and
            # log(max_patch_radius / 3) instead of clamping.
            log_min = math.log(self.min_scale)
            log_max = math.log(float(self.max_patch_radius) / 3.0)
            s = scale_param.data
            below = s < log_min
            above = s > log_max
            s[below] = 2 * log_min - s[below]
            s[above] = 2 * log_max - s[above]
            s.clamp_(log_min, log_max)

    def _create_split_children(
        self,
        selected: torch.Tensor,
        n_orig: int,
        N: int,
    ) -> dict[str, torch.Tensor]:
        """Create split children according to selected parameterization."""
        if self.parameterization == "cholesky":
            return self._create_split_children_cholesky(selected, n_orig, N)
        return self._create_split_children_rs(selected, n_orig, N)

    def _create_split_children_rs(
        self, selected: torch.Tensor, n_orig: int, N: int
    ) -> dict[str, torch.Tensor]:
        """Create RS-specific split children with rotated offsets."""
        # Sample offsets from parent's scale distribution
        scales_sel = self.get_scaling[:n_orig][selected]  # [K, 2]
        stds = scales_sel.repeat(N, 1)
        samples = torch.normal(
            mean=torch.zeros_like(stds), std=stds
        )

        # Rotate offset by parent rotation
        angles = self._rotation[:n_orig][selected, 0].repeat(N)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        rx = cos_a * samples[:, 0] - sin_a * samples[:, 1]
        ry = sin_a * samples[:, 0] + cos_a * samples[:, 1]
        offsets_pixel = torch.stack([rx, ry], dim=-1)

        parent_xy = self._xy[:n_orig][selected].detach().repeat(N, 1)

        return {
            "xy": parent_xy + offsets_pixel,
            "scaling": (
                self._scaling[:n_orig][selected].detach().repeat(N, 1)
                - math.log(0.8 * N)
            ),
            "rotation": (
                self._rotation[:n_orig][selected].detach().repeat(N, 1)
            ),
        }

    def _create_split_children_cholesky(
        self, selected: torch.Tensor, n_orig: int, N: int
    ) -> dict[str, torch.Tensor]:
        """Create Cholesky-specific split children."""
        l_diag_sel = torch.clamp(
            torch.exp(self._log_L_diag[:n_orig][selected]),
            min=self.min_scale,
            max=float(self.max_patch_radius) / 3.0,
        )
        l21_sel = self._L_offdiag[:n_orig][selected]

        l11 = l_diag_sel[:, 0]
        l22 = l_diag_sel[:, 1]
        l11_rep = l11.repeat(N)
        l22_rep = l22.repeat(N)
        l21_rep = l21_sel.repeat(N)

        z = torch.randn(l11_rep.shape[0], 2, device=l11.device)
        ox = l11_rep * z[:, 0]
        oy = l21_rep * z[:, 0] + l22_rep * z[:, 1]
        offsets_pixel = torch.stack([ox, oy], dim=-1)

        parent_xy = self._xy[:n_orig][selected].detach().repeat(N, 1)
        return {
            "xy": parent_xy + offsets_pixel,
            "log_L_diag": (
                self._log_L_diag[:n_orig][selected].detach().repeat(N, 1)
                - math.log(0.8 * N)
            ),
            "L_offdiag": (
                self._L_offdiag[:n_orig][selected].detach().repeat(N)
            ),
        }

    def _densify_and_clone(self, grads: torch.Tensor, n_orig: int, optimizer) -> None:
        """Polar mode halves amplitude only; Cartesian uses base behavior."""
        if self.weight_representation == "real_imag":
            super()._densify_and_clone(grads, n_orig, optimizer)
            return

        max_scale = self.get_scaling[:n_orig].max(dim=1).values
        selected = (grads.squeeze() >= self.densify_grad_threshold) & (
            max_scale <= self.densify_scale_threshold * max(self.output_H, self.output_W)
        )
        budget = self.max_gaussians - self.num_gaussians
        if not self._apply_budget(selected, grads, budget):
            return

        new_params = {}
        for name, attr in self._NAME_TO_ATTR.items():
            new_params[name] = getattr(self, attr)[selected].detach().clone()

        # Polar weights: halve amplitude only (col 0), keep phase (col 1).
        new_params["weight"][:, 0] *= 0.5
        self._weight.data[selected, 0] *= 0.5
        self._append_gaussians(optimizer, new_params)

    def _densify_and_split(
        self,
        grads: torch.Tensor,
        n_orig: int,
        optimizer,
        N: int = 2,
    ) -> None:
        """Polar mode splits amplitude; Cartesian uses base behavior."""
        if self.weight_representation == "real_imag":
            super()._densify_and_split(grads, n_orig, optimizer, N)
            return

        max_scale = self.get_scaling[:n_orig].max(dim=1).values
        selected = (grads.squeeze() >= self.densify_grad_threshold) & (
            max_scale > self.densify_scale_threshold * max(self.output_H, self.output_W)
        )
        budget = (self.max_gaussians - self.num_gaussians) // (N - 1)
        if not self._apply_budget(selected, grads, budget):
            return

        new_params = self._create_split_children(selected, n_orig, N)
        weight_data = self._weight[:n_orig][selected].detach().repeat(N, 1)
        weight_data[:, 0] /= N
        new_params["weight"] = weight_data
        self._append_gaussians(optimizer, new_params)

        prune_mask = torch.zeros(
            self.num_gaussians, dtype=torch.bool, device=self._xy.device
        )
        prune_mask[torch.where(selected)[0]] = True
        self._prune_by_mask(prune_mask, optimizer)

    def _prune(self, optimizer) -> None:
        """Polar mode prunes on amplitude, Cartesian uses base behavior.

        In MCMC mode, prune on raw weight magnitude (no opacity),
        matching the zhongtian reference.
        """
        if self.density_control == "mcmc":
            weight_mag = self._weight.detach().abs().sum(dim=-1)
            prune_mask = weight_mag < self.prune_weight_threshold
            if prune_mask.any():
                self._prune_by_mask(prune_mask, optimizer)
            return

        if self.weight_representation == "real_imag":
            super()._prune(optimizer)
            return

        amplitude = self._weight.detach()[:, 0].abs()
        prune_mask = amplitude < self.prune_weight_threshold
        if prune_mask.any():
            self._prune_by_mask(prune_mask, optimizer)


class CholeskyGaussianFieldModel(GaussianFieldModel):
    """Compatibility wrapper: fixed ``parameterization='cholesky'``."""

    def __init__(self, *args, **kwargs):
        kwargs["parameterization"] = "cholesky"
        super().__init__(*args, **kwargs)


class PolarGaussianFieldModel(GaussianFieldModel):
    """Compatibility wrapper: fixed ``weight_representation='amplitude_phase'``."""

    def __init__(self, *args, **kwargs):
        kwargs["weight_representation"] = "amplitude_phase"
        super().__init__(*args, **kwargs)
