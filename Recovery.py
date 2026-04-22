"""
Conventional far-field ptychography reconstruction.

Usage:
    python Recovery.py --config config/plant.yaml
"""

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import savemat
from tqdm import tqdm

from models.complex_inr import ComplexINRModel2D as FullModel
from models.gaussian_fields import ConventionalGSModel2D
from utils import (
    haar_wavelet_sparsity_loss,
    load_ptychography_data,
    quadratic_phase_probe,
    save_model_with_required_grad,
    scan_positions_to_pixels,
)

VALID_INIT_TYPES = ('initNone', 'initProbe', 'initQuadratic')
VALID_LOSS_TYPES = ('smooth_L1_loss', 'FD_loss', 'Poisson_likelihood_loss')
VALID_MODEL_TYPES = ('inr', 'gaussian_field')


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    cfg = {}
    for section in raw.values():
        if isinstance(section, dict):
            cfg.update(section)
    for k in ('wavelength', 'camera_length', 'camera_pixel_pitch', 'lr',
              'lr_decay_gamma', 'per_level_scale', 'first_omega_0',
              'hidden_omega_0', 'sparsity_weight', 'quadratic_focal_length'):
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])
    for k in ('gap', 'downsample_factor', 'iters', 'lr_decay_step',
              'n_levels', 'n_features_per_level', 'log2_hashmap_size',
              'base_resolution', 'hidden_features', 'hidden_layers',
              'vis_interval', 'batch_size', 'zoom_size'):
        if k in cfg and cfg[k] is not None:
            cfg[k] = int(cfg[k])

    init_type = cfg.get('init_type', 'initNone')
    if init_type not in VALID_INIT_TYPES:
        raise ValueError(f'init_type must be one of {VALID_INIT_TYPES}, got {init_type!r}')
    loss_type = cfg.get('loss_type', 'smooth_L1_loss')
    if loss_type not in VALID_LOSS_TYPES:
        raise ValueError(f'loss_type must be one of {VALID_LOSS_TYPES}, got {loss_type!r}')
    model_type = cfg.get('model_type', 'inr')
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f'model_type must be one of {VALID_MODEL_TYPES}, got {model_type!r}')
    return cfg


ap = argparse.ArgumentParser()
ap.add_argument('--config', type=str, default='config/plant.yaml')
args = ap.parse_args()
cfg = load_config(args.config)

gap = cfg.get('gap', 1)
wavelength = cfg['wavelength']
cameraLength = cfg['camera_length']
cameraPixelPitch = cfg['camera_pixel_pitch']

data = load_ptychography_data(
    data_dir=cfg['data_dir'],
    bundle_file=cfg.get('bundle_file'),
    location_file=cfg.get('location_file'),
    raw_file=cfg.get('raw_file'),
    probe_file=cfg.get('probe_file'),
    gt_file=cfg.get('gt_file'),
    init_probe_file=cfg.get('init_probe_file'),
    gap=gap,
)
xlocation = data['xlocation']
ylocation = data['ylocation']
probe = torch.from_numpy(np.asarray(data['probe'])).to('cuda')
inputFrames = data['imRaw']
gt_object = data.get('obj')
initProbe = (torch.from_numpy(np.asarray(data['initProbe'])).to('cuda')
             if 'initProbe' in data else None)

input = torch.tensor(inputFrames).float().to('cuda')
frames = input.shape[2]
print(f"Location shape: {xlocation.shape}")

tlX, tlY, brX, brY, dx, target_size = scan_positions_to_pixels(
    xlocation, ylocation, probe.shape,
    wavelength, cameraLength, cameraPixelPitch,
)
print("target canvas:", target_size)

batch_size = cfg.get('batch_size', 10)
cur_ds = cfg.get('downsample_factor', 2)
learning_rate = cfg.get('lr', 1e-3)
loss_type = cfg.get('loss_type', 'smooth_L1_loss')
lr_decay_step = cfg.get('lr_decay_step', 10000)
lr_decay_gamma = cfg.get('lr_decay_gamma', 0.1)
num_epochs = cfg.get('iters', 2000)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = cfg.get('use_amp', True)
sparsity_weight = cfg.get('sparsity_weight', 0.0)

n_levels = cfg.get('n_levels', 16)
n_features_per_level = cfg.get('n_features_per_level', 2)
log2_hashmap_size = cfg.get('log2_hashmap_size', 18)
base_resolution = cfg.get('base_resolution', 16)
per_level_scale = cfg.get('per_level_scale', 1.5)

first_omega_0 = cfg.get('first_omega_0', 10.0)
hidden_omega_0 = cfg.get('hidden_omega_0', 1.0)
hidden_features = cfg.get('hidden_features', 64)
hidden_layers = cfg.get('hidden_layers', 2)
trainable_omega0 = cfg.get('trainable_omega0', True)
use_residual = cfg.get('use_residual', False)
init_type = cfg.get('init_type', 'initNone')
quadratic_focal_length = cfg.get('quadratic_focal_length', 0.1)
model_type = cfg.get('model_type', 'inr')

vis_dir = cfg.get('result_dir', './result')
vis_interval = cfg.get('vis_interval', 100)
zoom_size = cfg.get('zoom_size', 300)
os.makedirs(vis_dir, exist_ok=True)
print("device:", device)
print(f"Results will be saved to: {vis_dir}")

if initProbe is not None and initProbe.dtype != torch.complex64:
    initProbe = initProbe.type(torch.complex64)

if init_type == 'initQuadratic':
    base = initProbe if initProbe is not None else probe
    probe_initial_torch = quadratic_phase_probe(
        base, dx, wavelength, quadratic_focal_length).to(device).to(torch.complex64)
elif init_type == 'initNone':
    probe_initial_torch = None
    use_residual = False
else:
    probe_initial_torch = (initProbe if initProbe is not None else probe).to(device)

probe_height, probe_width = probe.shape
if model_type == 'gaussian_field':
    gs_obj_cfg = cfg.get('gaussian_field', {}) or {}
    gs_probe_cfg = cfg.get('probe_gaussian_field', {}) or {}
    modelFn = ConventionalGSModel2D(
        output_width=target_size,
        output_height=target_size,
        downsample_factor=cur_ds,
        update_probe=True,
        probe_width=probe_width,
        probe_height=probe_height,
        use_residual=use_residual,
        object_initial=None,
        probe_initial=probe_initial_torch,
        object_num_initial_gaussians=int(gs_obj_cfg.get('num_initial_gaussians', 30000)),
        probe_num_initial_gaussians=int(gs_probe_cfg.get('num_initial_gaussians', 15000)),
        parameterization=gs_obj_cfg.get('parameterization', 'cholesky'),
        weight_representation=gs_obj_cfg.get('weight_representation', 'real_imag'),
        phase_init_std=float(gs_obj_cfg.get('phase_init_std', 0.3)),
        object_densify_grad_threshold=float(gs_obj_cfg.get('densify_grad_threshold', 5e-6)),
        probe_densify_grad_threshold=float(gs_probe_cfg.get('densify_grad_threshold', 5e-6)),
        object_densify_interval=int(gs_obj_cfg.get('densify_interval', 500)),
        probe_densify_interval=int(gs_probe_cfg.get('densify_interval', 300)),
        object_densify_until_step=int(gs_obj_cfg.get('densify_until_step', 15000)),
        probe_densify_until_step=int(gs_probe_cfg.get('densify_until_step', 10000)),
        object_max_gaussians=int(gs_obj_cfg.get('max_gaussians', 80000)),
        probe_max_gaussians=int(gs_probe_cfg.get('max_gaussians', 60000)),
        object_init_scale=float(gs_obj_cfg.get('init_scale', 5.0)),
        probe_init_scale=float(gs_probe_cfg.get('init_scale', 1.0)),
        object_min_scale=float(gs_obj_cfg.get('min_scale', 0.5)),
        probe_min_scale=float(gs_probe_cfg.get('min_scale', 0.0)),
        max_patch_radius=int(gs_obj_cfg.get('max_patch_radius', 16)),
        object_density_control=str(gs_obj_cfg.get('density_control', 'adc')),
        probe_density_control=str(gs_probe_cfg.get('density_control', 'adc')),
        object_mcmc_grow_rate=float(gs_obj_cfg.get('mcmc_grow_rate', 0.05)),
        probe_mcmc_grow_rate=float(gs_probe_cfg.get('mcmc_grow_rate', 0.05)),
        object_mcmc_relocation_fraction=float(gs_obj_cfg.get('mcmc_relocation_fraction', 0.05)),
        probe_mcmc_relocation_fraction=float(gs_probe_cfg.get('mcmc_relocation_fraction', 0.05)),
        object_mcmc_noise_lr_scale=float(gs_obj_cfg.get('mcmc_noise_lr_scale', 1.0)),
        probe_mcmc_noise_lr_scale=float(gs_probe_cfg.get('mcmc_noise_lr_scale', 1.0)),
    ).to(device)
else:
    modelFn = FullModel(
        output_width=target_size,
        output_height=target_size,
        downsample_factor=cur_ds,
        update_probe=True,
        probe_width=probe_width,
        probe_height=probe_height,
        use_residual=use_residual,
        object_initial=None,
        probe_initial=probe_initial_torch,
        n_levels=n_levels,
        n_features_per_level=n_features_per_level,
        log2_hashmap_size=log2_hashmap_size,
        base_resolution=base_resolution,
        per_level_scale=per_level_scale,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        trainable_omega0=trainable_omega0,
    ).to(device)

central_pixel = target_size // 2


# ---- training loop (mirrors legacy/plant/Recovery.py; do not restructure) ---
t = tqdm(range(num_epochs))
if model_type == 'gaussian_field':
    param_groups = list(modelFn.object_model.get_param_groups(learning_rate))
    if modelFn.probe_model is not None:
        for g in modelFn.probe_model.get_param_groups(learning_rate):
            g = dict(g)
            g['name'] = 'probe_' + g.get('name', 'unknown')
            param_groups.append(g)
    optimizer = torch.optim.Adam(param_groups, lr=learning_rate)
else:
    optimizer = torch.optim.Adam(
        lr=learning_rate, params=filter(lambda p: p.requires_grad, modelFn.parameters()))
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

for epoch in t:
    perm = torch.randperm(frames, device=device)

    with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
        for batch_start in range(0, frames, batch_size):
            batch_idx = perm[batch_start: batch_start + batch_size]

            modelFn.zero_grad()
            optimizer.zero_grad()

            object_complex, probe_complex = modelFn()
            if probe_complex is None:
                raise RuntimeError("update_probe must be True for conventional ptychography")
            probe_norm = torch.sqrt((probe_complex.abs() ** 2).sum())
            probe_complex = probe_complex / probe_norm

            if probe_complex.dim() > 2:
                probe_complex = probe_complex.squeeze()
            if object_complex.dim() > 2:
                object_complex = object_complex.squeeze()

            absP2 = probe_complex.abs() ** 2
            Mp, Np = absP2.shape
            tot = absP2.sum()
            cp = torch.trunc(torch.tensor([Mp, Np], device=probe_complex.device) / 2 -
                             torch.tensor([Mp, Np], device=probe_complex.device) *
                             torch.stack((absP2.sum(1).cumsum(0).mean(),
                                          absP2.sum(0).cumsum(0).mean())) / tot + 1).long()
            probe_complex = (torch.roll(probe_complex, (-cp[0], -cp[1]), (0, 1))
                             if cp.any() else probe_complex)

            loss_batch = 0.0
            for id in batch_idx:
                oI_cap = input[:, :, id]
                currentEW = probe_complex * object_complex[tlY[id]:brY[id], tlX[id]:brX[id]]
                fft_EW = torch.fft.fftshift(torch.fft.fft2(currentEW), dim=(-2, -1))
                oI_sub = torch.abs(fft_EW) ** 2

                if loss_type == 'smooth_L1_loss':
                    loss = F.smooth_l1_loss(oI_sub, oI_cap)
                elif loss_type == 'FD_loss':
                    eps = 1e-8
                    dW = torch.sqrt(oI_cap + eps) - torch.sqrt(oI_sub + eps)
                    RxdW = torch.cat((dW[:, 1:] - dW[:, :-1], dW[:, :1] - dW[:, -1:]), dim=1)
                    RydW = torch.cat((dW[1:, :] - dW[:-1, :], dW[:1, :] - dW[-1:, :]), dim=0)
                    gradient_magnitude = torch.sqrt(RxdW ** 2 + RydW ** 2 + eps)
                    loss = gradient_magnitude.mean()
                elif loss_type == 'Poisson_likelihood_loss':
                    eps = 1e-8
                    loss = (oI_sub - oI_cap * torch.log(oI_sub + eps)).mean()

                loss_batch += loss

            sparsity_loss = haar_wavelet_sparsity_loss(object_complex)
            sparsity_loss_total = sparsity_weight * sparsity_loss

            data_loss_val = loss_batch.detach().item()
            loss_batch = loss_batch + sparsity_loss_total

            retain = (batch_start + batch_size) < frames
            loss_batch.backward(retain_graph=retain)
            if model_type == 'gaussian_field':
                modelFn.object_model.accumulate_gradients()
                if modelFn.probe_model is not None:
                    modelFn.probe_model.accumulate_gradients()
            optimizer.step()
            if model_type == 'gaussian_field':
                modelFn.object_model.densification_step(optimizer)
                if modelFn.probe_model is not None:
                    modelFn.probe_model.densification_step(optimizer)
                # MCMC-only: SGLD noise injection on xy + scale params.
                # sgld_noise_step internally no-ops when density_control != "mcmc".
                def _lr_by_name(opt, *names):
                    for g in opt.param_groups:
                        if g.get('name') in names:
                            return float(g['lr'])
                    return None
                obj_lr_xy = _lr_by_name(optimizer, 'xy')
                obj_lr_scale = _lr_by_name(optimizer, 'log_L_diag', 'scaling')
                if obj_lr_xy is not None and obj_lr_scale is not None:
                    modelFn.object_model.field.sgld_noise_step(obj_lr_xy, obj_lr_scale)
                if modelFn.probe_model is not None:
                    pr_lr_xy = _lr_by_name(optimizer, 'probe_xy')
                    pr_lr_scale = _lr_by_name(optimizer, 'probe_log_L_diag', 'probe_scaling')
                    if pr_lr_xy is not None and pr_lr_scale is not None:
                        modelFn.probe_model.field.sgld_noise_step(pr_lr_xy, pr_lr_scale)

            sparsity_loss_val = (sparsity_loss_total.detach().item()
                                 if torch.is_tensor(sparsity_loss_total) else 0.0)
            t.set_postfix(
                TotalLoss=f"{loss_batch.detach().item():.4e}",
                DataLoss=f"{data_loss_val:.4e}",
                SparsityLoss=f"{sparsity_loss_val:.4e}",
            )
    scheduler.step()

    if epoch % vis_interval == 0 or epoch == num_epochs - 1:
        amplitude = torch.abs(object_complex.squeeze()).cpu().detach().numpy()
        phase = torch.angle(object_complex.squeeze()).cpu().detach().numpy()
        probe_np = probe_complex.cpu().detach().numpy()

        half_zoom = zoom_size // 2
        obj_slice = slice(central_pixel - half_zoom, central_pixel + half_zoom)
        data_to_plot = [
            (amplitude[obj_slice, obj_slice], "Object Amplitude"),
            (phase[obj_slice, obj_slice], "Object Phase"),
            (np.abs(probe_np), "Probe Amplitude"),
            (np.angle(probe_np), "Probe Phase"),
        ]

        fig, axs = plt.subplots(1, 4, figsize=(24, 5), dpi=150)
        for ax, (d, title) in zip(axs, data_to_plot):
            im = ax.imshow(d, cmap="gray")
            ax.axis("image")
            ax.set_title(title)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/e_{epoch}.png", dpi=150)
        plt.close(fig)
        savemat(f"{vis_dir}/result.mat", {
            'object_complex': object_complex.squeeze().cpu().detach().numpy(),
            'probe_complex': probe_complex.squeeze().cpu().detach().numpy(),
        })

        save_path = os.path.join(vis_dir, 'trained_models.pth')
        save_model_with_required_grad(modelFn, save_path)
