"""
Conventional far-field ptychography reconstruction.

Flat script aligned with legacy/plant/Recovery.py. Hyperparameters are
loaded from a YAML config; the training loop is left untouched.

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
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat, savemat
from tqdm import tqdm

from models.complex_inr import ComplexINRModel2D as FullModel
from utils import haar_wavelet_sparsity_loss, save_model_with_required_grad


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
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
    return cfg


ap = argparse.ArgumentParser()
ap.add_argument('--config', type=str, default='config/plant.yaml')
args = ap.parse_args()
cfg = load_config(args.config)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
gap = cfg.get('gap', 1)
wavelength = cfg['wavelength']
cameraLength = cfg['camera_length']
cameraPixelPitch = cfg['camera_pixel_pitch']
data_dir = cfg['data_dir']
bundle_file = cfg.get('bundle_file')

if bundle_file:
    data = loadmat(os.path.join(data_dir, bundle_file))
    xlocation = data['xlocation'][:, 0::gap].squeeze()
    ylocation = data['ylocation'][:, 0::gap].squeeze()
    probe = torch.from_numpy(data['probe']).to('cuda')
    inputFrames = data['imRaw']
    gt_object = data.get('obj', None)
    initProbe = (torch.from_numpy(data['initProbe']).to('cuda')
                 if 'initProbe' in data else None)
else:
    loc = loadmat(os.path.join(data_dir, cfg['location_file']))
    xlocation = np.asarray(loc['xlocation']).squeeze()
    ylocation = np.asarray(loc['ylocation']).squeeze()
    if xlocation.ndim == 0:
        raise RuntimeError('xlocation must be at least 1-D')
    xlocation = xlocation[::gap]
    ylocation = ylocation[::gap]
    probe = torch.from_numpy(
        loadmat(os.path.join(data_dir, cfg['probe_file']))['probe']).to('cuda')
    inputFrames = loadmat(os.path.join(data_dir, cfg['raw_file']))['imRaw']
    gt_object = (loadmat(os.path.join(data_dir, cfg['gt_file']))['obj']
                 if cfg.get('gt_file') else None)
    initProbe = (torch.from_numpy(
        loadmat(os.path.join(data_dir, cfg['init_probe_file']))['initProbe']
    ).to('cuda') if cfg.get('init_probe_file') else None)

input = torch.tensor(inputFrames[:, :, 0::gap]).float().to('cuda')
frames = input.shape[2]
print(f"Location shape: {xlocation.shape}")


# -----------------------------------------------------------------------------
# Pixel geometry
# -----------------------------------------------------------------------------
x_pixel_positions = xlocation - np.min(xlocation)
y_pixel_positions = ylocation - np.min(ylocation)

M, N = probe.shape
dx = wavelength * cameraLength / (np.array([M, N]) * cameraPixelPitch)
tlY = np.round(y_pixel_positions / dx[0]).astype(int)
tlX = np.round(x_pixel_positions / dx[1]).astype(int)
brY = tlY + M
brX = tlX + N

obj = np.ones((np.max(brY), np.max(brX)))
imSizeX, imSizeY = obj.shape
print('imSizeX:', imSizeX)
print('imSizeY:', imSizeY)
max_size = max(imSizeX, imSizeY)
target_size = math.ceil(max_size / 6) * 6
print("target canvas:", target_size)


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
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

vis_dir = cfg.get('result_dir', './result')
vis_interval = cfg.get('vis_interval', 100)
zoom_size = cfg.get('zoom_size', 300)
os.makedirs(vis_dir, exist_ok=True)
print("device:", device)
print(f"Results will be saved to: {vis_dir}")


# -----------------------------------------------------------------------------
# Probe initial (for residual connection)
# -----------------------------------------------------------------------------
if initProbe is not None and initProbe.dtype != torch.complex64:
    initProbe = initProbe.type(torch.complex64)

probe_height, probe_width = probe.shape

if init_type == 'initQuadratic':
    xp = (np.arange(N, dtype=np.float32) - N / 2) * dx[1]
    yp = (np.arange(M, dtype=np.float32) - M / 2) * dx[0]
    Xp, Yp = np.meshgrid(xp, yp)
    quad_phase = np.pi / (wavelength * quadratic_focal_length) * (Xp ** 2 + Yp ** 2)
    base = initProbe.cpu().numpy() if initProbe is not None else probe.cpu().numpy()
    init_probe_np = np.exp(1j * quad_phase).astype(np.complex64) * base.astype(np.complex64)
    probe_initial_torch = torch.from_numpy(init_probe_np).to(device)
elif init_type == 'initNone':
    probe_initial_torch = None
    use_residual = False
else:  # initProbe
    probe_initial_torch = initProbe.to(device) if initProbe is not None else probe.to(device)

if probe_initial_torch is not None and probe_initial_torch.dtype != torch.complex64:
    probe_initial_torch = probe_initial_torch.type(torch.complex64)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
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

central_pixel = imSizeX // 2


# -----------------------------------------------------------------------------
# Training loop (unchanged from legacy Recovery.py)
# -----------------------------------------------------------------------------
t = tqdm(range(num_epochs))
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
                raise RuntimeError("当前配置需要启用 probe 分支，请确保 update_probe=True")
            probe_norm = torch.sqrt((probe_complex.abs() ** 2).sum())
            probe_complex = probe_complex / probe_norm

            if probe_complex.dim() > 2:
                probe_complex = probe_complex.squeeze()
            if object_complex.dim() > 2:
                object_complex = object_complex.squeeze()

            # Probe 中心对齐
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
            optimizer.step()

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
