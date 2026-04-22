from __future__ import annotations

import torch
import torch.nn as nn

from gscp.models import GaussianFieldModel


class _GaussianFieldComponent2D(nn.Module):
    """Thin wrapper around the vendored GSCP Gaussian field model."""

    def __init__(
        self,
        *,
        output_width: int,
        output_height: int,
        downsample_factor: int,
        num_initial_gaussians: int,
        initial_field: torch.Tensor | None = None,
        use_residual: bool = False,
        residual_scale_init: float = 1e-8,
        parameterization: str = "cholesky",
        weight_representation: str = "real_imag",
        phase_init_std: float = 0.3,
        densify_grad_threshold: float = 5e-6,
        densify_interval: int = 500,
        densify_until_step: int = 10000,
        max_gaussians: int = 500000,
        init_scale: float = 5.0,
        min_scale: float = 0.5,
        max_patch_radius: int = 16,
        density_control: str = "adc",
        mcmc_grow_rate: float = 0.05,
        mcmc_relocation_fraction: float = 0.05,
        mcmc_noise_lr_scale: float = 1.0,
        label: str = "",
    ) -> None:
        super().__init__()
        if output_width % downsample_factor != 0 or output_height % downsample_factor != 0:
            raise ValueError(
                "output size must be divisible by downsample_factor for GaussianFieldModel"
            )

        has_residual = use_residual and initial_field is not None
        model_initial = None if has_residual else initial_field

        self.field = GaussianFieldModel(
            output_H=output_height,
            output_W=output_width,
            downsample_factor=downsample_factor,
            num_initial_gaussians=num_initial_gaussians,
            object_recovery_initial=model_initial,
            densify_grad_threshold=densify_grad_threshold,
            densify_interval=densify_interval,
            densify_until_step=densify_until_step,
            max_gaussians=max_gaussians,
            init_scale=init_scale,
            min_scale=min_scale,
            max_patch_radius=max_patch_radius,
            parameterization=parameterization,
            weight_representation=weight_representation,
            phase_init_std=phase_init_std,
            density_control=density_control,
            mcmc_grow_rate=mcmc_grow_rate,
            mcmc_relocation_fraction=mcmc_relocation_fraction,
            mcmc_noise_lr_scale=mcmc_noise_lr_scale,
            label=label,
        )

        self.use_residual = has_residual
        if has_residual:
            initial = initial_field.detach().clone().to(torch.complex64)
            self.register_buffer("initial_field", initial)
            self.residual_scale = nn.Parameter(
                torch.tensor(float(residual_scale_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "initial_field",
                torch.zeros(0, dtype=torch.complex64),
                persistent=False,
            )
            self.residual_scale = None

    def forward(self) -> torch.Tensor:
        field = self.field()
        if self.use_residual:
            field = self.initial_field + self.residual_scale * field
        return field

    def get_param_groups(self, base_lr: float):
        groups = list(self.field.get_param_groups(base_lr))
        if self.residual_scale is not None:
            groups.append(
                {
                    "params": [self.residual_scale],
                    "lr": base_lr,
                    "name": "residual_scale",
                }
            )
        return groups

    def densification_step(self, optimizer: torch.optim.Optimizer) -> None:
        self.field.densification_step(optimizer)

    def accumulate_gradients(self) -> None:
        self.field.accumulate_gradients()

    @property
    def num_gaussians(self) -> int:
        return self.field.num_gaussians


class ObjectGaussianField2D(_GaussianFieldComponent2D):
    def __init__(
        self,
        *,
        output_width: int,
        output_height: int,
        downsample_factor: int = 2,
        num_initial_gaussians: int = 30000,
        initial_field: torch.Tensor | None = None,
        use_residual: bool = False,
        residual_scale_init: float = 1e-8,
        parameterization: str = "cholesky",
        weight_representation: str = "real_imag",
        phase_init_std: float = 0.3,
        densify_grad_threshold: float = 5e-6,
        densify_interval: int = 500,
        densify_until_step: int = 15000,
        max_gaussians: int = 80000,
        init_scale: float = 5.0,
        min_scale: float = 0.5,
        max_patch_radius: int = 16,
        density_control: str = "adc",
        mcmc_grow_rate: float = 0.05,
        mcmc_relocation_fraction: float = 0.05,
        mcmc_noise_lr_scale: float = 1.0,
    ) -> None:
        super().__init__(
            output_width=output_width,
            output_height=output_height,
            downsample_factor=downsample_factor,
            num_initial_gaussians=num_initial_gaussians,
            initial_field=initial_field,
            use_residual=use_residual,
            residual_scale_init=residual_scale_init,
            parameterization=parameterization,
            weight_representation=weight_representation,
            phase_init_std=phase_init_std,
            densify_grad_threshold=densify_grad_threshold,
            densify_interval=densify_interval,
            densify_until_step=densify_until_step,
            max_gaussians=max_gaussians,
            init_scale=init_scale,
            min_scale=min_scale,
            max_patch_radius=max_patch_radius,
            density_control=density_control,
            mcmc_grow_rate=mcmc_grow_rate,
            mcmc_relocation_fraction=mcmc_relocation_fraction,
            mcmc_noise_lr_scale=mcmc_noise_lr_scale,
            label="object",
        )


class ProbeGaussianField2D(_GaussianFieldComponent2D):
    def __init__(
        self,
        *,
        output_width: int,
        output_height: int,
        downsample_factor: int = 1,
        num_initial_gaussians: int = 15000,
        initial_field: torch.Tensor | None = None,
        use_residual: bool = False,
        residual_scale_init: float = 1e-8,
        parameterization: str = "cholesky",
        weight_representation: str = "real_imag",
        phase_init_std: float = 0.3,
        densify_grad_threshold: float = 5e-6,
        densify_interval: int = 300,
        densify_until_step: int = 10000,
        max_gaussians: int = 60000,
        init_scale: float = 1.0,
        min_scale: float = 0.0,
        max_patch_radius: int = 16,
        density_control: str = "adc",
        mcmc_grow_rate: float = 0.05,
        mcmc_relocation_fraction: float = 0.05,
        mcmc_noise_lr_scale: float = 1.0,
    ) -> None:
        super().__init__(
            output_width=output_width,
            output_height=output_height,
            downsample_factor=downsample_factor,
            num_initial_gaussians=num_initial_gaussians,
            initial_field=initial_field,
            use_residual=use_residual,
            residual_scale_init=residual_scale_init,
            parameterization=parameterization,
            weight_representation=weight_representation,
            phase_init_std=phase_init_std,
            densify_grad_threshold=densify_grad_threshold,
            densify_interval=densify_interval,
            densify_until_step=densify_until_step,
            max_gaussians=max_gaussians,
            init_scale=init_scale,
            min_scale=min_scale,
            max_patch_radius=max_patch_radius,
            density_control=density_control,
            mcmc_grow_rate=mcmc_grow_rate,
            mcmc_relocation_fraction=mcmc_relocation_fraction,
            mcmc_noise_lr_scale=mcmc_noise_lr_scale,
            label="probe",
        )


class ConventionalGSModel2D(nn.Module):
    """Combined object/probe Gaussian-field model with an INR-like API."""

    def __init__(
        self,
        output_width: int,
        output_height: int,
        downsample_factor: int,
        update_probe: bool = True,
        probe_width: int | None = None,
        probe_height: int | None = None,
        use_residual: bool = False,
        object_initial: torch.Tensor | None = None,
        probe_initial: torch.Tensor | None = None,
        object_num_initial_gaussians: int = 30000,
        probe_num_initial_gaussians: int = 15000,
        parameterization: str = "cholesky",
        weight_representation: str = "real_imag",
        phase_init_std: float = 0.3,
        object_densify_grad_threshold: float = 5e-6,
        probe_densify_grad_threshold: float = 5e-6,
        object_densify_interval: int = 500,
        probe_densify_interval: int = 300,
        object_densify_until_step: int = 15000,
        probe_densify_until_step: int = 10000,
        object_max_gaussians: int = 80000,
        probe_max_gaussians: int = 60000,
        object_init_scale: float = 5.0,
        probe_init_scale: float = 1.0,
        object_min_scale: float = 0.5,
        probe_min_scale: float = 0.0,
        max_patch_radius: int = 16,
        residual_scale_init: float = 1e-8,
        object_density_control: str = "adc",
        probe_density_control: str = "adc",
        object_mcmc_grow_rate: float = 0.05,
        probe_mcmc_grow_rate: float = 0.05,
        object_mcmc_relocation_fraction: float = 0.05,
        probe_mcmc_relocation_fraction: float = 0.05,
        object_mcmc_noise_lr_scale: float = 1.0,
        probe_mcmc_noise_lr_scale: float = 1.0,
    ) -> None:
        super().__init__()
        probe_width = probe_width if probe_width is not None else output_width
        probe_height = probe_height if probe_height is not None else output_height

        self.update_probe = update_probe
        self.object_model = ObjectGaussianField2D(
            output_width=output_width,
            output_height=output_height,
            downsample_factor=downsample_factor,
            num_initial_gaussians=object_num_initial_gaussians,
            initial_field=object_initial,
            use_residual=use_residual,
            residual_scale_init=residual_scale_init,
            parameterization=parameterization,
            weight_representation=weight_representation,
            phase_init_std=phase_init_std,
            densify_grad_threshold=object_densify_grad_threshold,
            densify_interval=object_densify_interval,
            densify_until_step=object_densify_until_step,
            max_gaussians=object_max_gaussians,
            init_scale=object_init_scale,
            min_scale=object_min_scale,
            max_patch_radius=max_patch_radius,
            density_control=object_density_control,
            mcmc_grow_rate=object_mcmc_grow_rate,
            mcmc_relocation_fraction=object_mcmc_relocation_fraction,
            mcmc_noise_lr_scale=object_mcmc_noise_lr_scale,
        )
        self.probe_model = (
            ProbeGaussianField2D(
                output_width=probe_width,
                output_height=probe_height,
                downsample_factor=1,
                num_initial_gaussians=probe_num_initial_gaussians,
                initial_field=probe_initial,
                use_residual=use_residual,
                residual_scale_init=residual_scale_init,
                parameterization=parameterization,
                weight_representation=weight_representation,
                phase_init_std=phase_init_std,
                densify_grad_threshold=probe_densify_grad_threshold,
                densify_interval=probe_densify_interval,
                densify_until_step=probe_densify_until_step,
                max_gaussians=probe_max_gaussians,
                init_scale=probe_init_scale,
                min_scale=probe_min_scale,
                max_patch_radius=max_patch_radius,
                density_control=probe_density_control,
                mcmc_grow_rate=probe_mcmc_grow_rate,
                mcmc_relocation_fraction=probe_mcmc_relocation_fraction,
                mcmc_noise_lr_scale=probe_mcmc_noise_lr_scale,
            )
            if update_probe
            else None
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        object_complex = self.object_model()
        probe_complex = self.probe_model() if self.probe_model is not None else None
        return object_complex, probe_complex

