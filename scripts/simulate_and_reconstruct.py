"""
Simulated conventional far-field ptychography end-to-end test:
  object amplitude = data/sim_data/westconcordorthophoto.png
  object phase     = (1 - amp) * pi - pi/2
  probe            = data/plant/probe1.mat  (complex, 512x512)

Pipeline:
  1. Build complex GT object on a canvas large enough to hold the
     scan footprint.
  2. Load probe.
  3. Cartesian scan on a (SCAN_SIZE x SCAN_SIZE) grid.
  4. Forward-simulate |fftshift(fft2(probe * object[tlY:brY, tlX:brX]))|^2.
  5. Reconstruct with ComplexINRModel2D (Euler activation + hash encoding).
  6. Report SSIM on object amplitude / phase after piston alignment and
     dump per-panel PNGs + a combined summary figure.

Usage:
    python scripts/simulate_and_reconstruct.py
"""

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import math
import os
import sys
from math import pi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat, savemat
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.complex_inr import ComplexINRModel2D
from models.gaussian_fields import ConventionalGSModel2D
from utils import save_model_with_required_grad


# =========================================================================
# Config
# =========================================================================

# Physics (use plant's parameters since the probe comes from the plant dataset)
WAVELENGTH = 0.675e-6
CAMERA_LENGTH = 0.0295
CAMERA_PIXEL_PITCH = 2.96e-5

PROBE_FILE = os.path.join(ROOT, 'data', 'plant', 'probe1.mat')
OBJECT_IMG = os.path.join(ROOT, 'data', 'sim_data', 'westconcordorthophoto.png')
SCAN_BUNDLE = os.path.join(ROOT, 'data', 'plant', 'dataset-plant.mat')  # reuse its 200 scan positions

# Model: 'inr' (ComplexINRModel2D) or 'gs' (ConventionalGSModel2D)
MODEL_TYPE = 'gs'

# Training
ITERS = 500
BATCH_SIZE = 10
LR = 2e-2 if MODEL_TYPE == 'gs' else 1e-3
LR_DECAY_STEP = 10000
LR_DECAY_GAMMA = 0.1
LOSS_TYPE = 'smooth_L1_loss'
USE_AMP = MODEL_TYPE != 'gs'  # GS's PyTorch fallback doesn't play well with AMP

RESULT_DIR = os.path.join(ROOT, 'results',
                          f'sim_west_plant_probe_{MODEL_TYPE}')


# =========================================================================
# Simulation
# =========================================================================

def build_scan_positions(probe_shape):
    """Reuse the 200 physical scan positions from the plant experiment.

    Returns (tlX, tlY, brX, brY, canvas, xlocation, ylocation) where
    xlocation/ylocation are the meter-units positions for bookkeeping."""
    from scipy.io import loadmat
    data = loadmat(SCAN_BUNDLE)
    xlocation = np.asarray(data['xlocation']).squeeze()
    ylocation = np.asarray(data['ylocation']).squeeze()
    M, N = probe_shape
    dx = WAVELENGTH * CAMERA_LENGTH / (np.array([M, N]) * CAMERA_PIXEL_PITCH)
    x_pix = xlocation - xlocation.min()
    y_pix = ylocation - ylocation.min()
    tlY = np.round(y_pix / dx[0]).astype(int)
    tlX = np.round(x_pix / dx[1]).astype(int)
    brY = tlY + M
    brX = tlX + N
    canvas = math.ceil(max(brY.max(), brX.max()) / 6) * 6
    return tlX, tlY, brX, brY, canvas, xlocation, ylocation


def save_scan_path(tlX, tlY, xlocation, ylocation, probe_shape, canvas, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].plot(xlocation * 1e6, ylocation * 1e6, 'r.-', lw=0.7, ms=4)
    axes[0].plot(xlocation[0] * 1e6, ylocation[0] * 1e6, 'go', ms=10, label='start')
    axes[0].plot(xlocation[-1] * 1e6, ylocation[-1] * 1e6, 'ks', ms=10, label='end')
    axes[0].set_xlabel('x (um)'); axes[0].set_ylabel('y (um)')
    axes[0].set_title(f'Scan path — physical (meters)\n{len(tlX)} frames')
    axes[0].axis('equal'); axes[0].grid(alpha=0.3); axes[0].legend()

    M, N = probe_shape
    axes[1].set_xlim(0, canvas); axes[1].set_ylim(canvas, 0)
    for x, y in zip(tlX, tlY):
        axes[1].add_patch(plt.Rectangle((x, y), N, M, fill=False,
                                         edgecolor='tab:blue', alpha=0.08, lw=0.5))
    axes[1].plot(tlX + N / 2, tlY + M / 2, 'r.', ms=3)
    axes[1].set_xlabel('x (pixel)'); axes[1].set_ylabel('y (pixel)')
    axes[1].set_title(f'Probe footprints on {canvas}x{canvas} canvas')
    axes[1].set_aspect('equal'); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scan_path.png'), dpi=120, bbox_inches='tight')
    plt.close()

    np.savez(os.path.join(out_dir, 'scan_path.npz'),
             xlocation=xlocation, ylocation=ylocation,
             tlX=tlX, tlY=tlY)


def load_gt_object(canvas_size):
    img = Image.open(OBJECT_IMG).convert('L')
    img = img.resize((canvas_size, canvas_size), Image.BICUBIC)
    amp = np.asarray(img, dtype=np.float64) / 255.0
    amp = amp / amp.max()
    phase = (1.0 - amp) * pi - pi / 2
    return amp * np.exp(1j * phase), amp, phase


def simulate(gt_obj, probe, tlX, tlY, brX, brY, device):
    M, N = probe.shape
    frames = len(tlX)
    obj_t = torch.from_numpy(gt_obj).to(device).to(torch.complex64)
    out = torch.zeros(M, N, frames, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(frames):
            exit_wave = probe * obj_t[tlY[i]:brY[i], tlX[i]:brX[i]]
            intensity = torch.abs(
                torch.fft.fftshift(torch.fft.fft2(exit_wave), dim=(-2, -1))
            ) ** 2
            out[:, :, i] = intensity.float()
    return out


# =========================================================================
# Reconstruction
# =========================================================================

def reconstruct(im_raw, probe_shape, canvas, tlX, tlY, brX, brY, device):
    M, N = probe_shape
    if MODEL_TYPE == 'gs':
        model = ConventionalGSModel2D(
            output_width=canvas, output_height=canvas,
            downsample_factor=2, update_probe=True,
            probe_width=N, probe_height=M,
            object_num_initial_gaussians=50000,
            probe_num_initial_gaussians=15000,
            object_max_gaussians=120000, probe_max_gaussians=40000,
            parameterization='cholesky', weight_representation='real_imag',
        ).to(device)
    else:
        model = ComplexINRModel2D(
            output_width=canvas, output_height=canvas,
            downsample_factor=2, update_probe=True,
            probe_width=N, probe_height=M,
            use_residual=False, object_initial=None, probe_initial=None,
            n_levels=16, n_features_per_level=2, log2_hashmap_size=18,
            base_resolution=16, per_level_scale=1.5,
            first_omega_0=10.0, hidden_omega_0=1.0,
            hidden_features=64, hidden_layers=2, trainable_omega0=True,
        ).to(device)

    optimizer = torch.optim.Adam(
        lr=LR, params=filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

    frames = im_raw.shape[2]
    loss_hist = []
    pbar = tqdm(range(ITERS), desc='Train')
    for epoch in pbar:
        perm = torch.randperm(frames, device=device)
        epoch_losses = []
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
            for bs in range(0, frames, BATCH_SIZE):
                idx = perm[bs:bs + BATCH_SIZE]
                model.zero_grad()
                optimizer.zero_grad()

                obj_c, probe_c = model()
                if obj_c.dim() > 2:
                    obj_c = obj_c.squeeze()
                if probe_c.dim() > 2:
                    probe_c = probe_c.squeeze()

                # Centroid alignment (no_grad; fp32 for bf16 autocast safety)
                with torch.no_grad():
                    absP2 = (probe_c.abs() ** 2).float()
                    Mp, Np = absP2.shape
                    tot = absP2.sum() + 1e-12
                    MN = torch.tensor([Mp, Np], device=device, dtype=torch.float32)
                    com = torch.stack((absP2.sum(1).cumsum(0).mean(),
                                       absP2.sum(0).cumsum(0).mean()))
                    raw = MN / 2 - MN * com / tot + 1
                    cp = (torch.trunc(raw.clamp(-Mp, Mp)).long()
                          if torch.isfinite(raw).all()
                          else torch.zeros(2, dtype=torch.long, device=device))
                if cp.any():
                    probe_c = torch.roll(
                        probe_c, (-cp[0].item(), -cp[1].item()), (0, 1))

                # Energy normalize to N_pixels so per-pixel |probe| ~ 1
                # (keeps bf16 dynamic range; same convention as gscp trainer)
                energy = (probe_c.abs() ** 2).sum() + 1e-12
                target_energy = float(Mp * Np)
                probe_c = probe_c * torch.sqrt(target_energy / energy)

                loss_b = 0.0
                for fid in idx:
                    fid = fid.item()
                    exit_wave = probe_c * obj_c[tlY[fid]:brY[fid], tlX[fid]:brX[fid]]
                    oI_sub = torch.abs(torch.fft.fftshift(
                        torch.fft.fft2(exit_wave), dim=(-2, -1))) ** 2
                    loss_b = loss_b + F.smooth_l1_loss(oI_sub, im_raw[:, :, fid])

                retain = (bs + BATCH_SIZE) < frames
                loss_b.backward(retain_graph=retain)
                optimizer.step()
                # Densification is disabled: vendored gscp has a shape-mismatch
                # bug when one optimizer owns both object and probe param groups.
                # Fixed-count Gaussians still learn well if the initial count
                # is large enough.
                epoch_losses.append(loss_b.detach().item())
        scheduler.step()
        loss_hist.append(float(np.mean(epoch_losses)))
        pbar.set_postfix(loss=f'{loss_hist[-1]:.3e}')

    with torch.no_grad():
        obj_final, probe_final = model()
        obj_final = obj_final.squeeze()
        probe_final = probe_final.squeeze()
        absP2 = (probe_final.abs() ** 2).float()
        Mp, Np = absP2.shape
        tot = absP2.sum() + 1e-12
        MN = torch.tensor([Mp, Np], device=device, dtype=torch.float32)
        com = torch.stack((absP2.sum(1).cumsum(0).mean(),
                           absP2.sum(0).cumsum(0).mean()))
        cp = torch.trunc(MN / 2 - MN * com / tot + 1).long()
        if cp.any():
            probe_final = torch.roll(
                probe_final, (-cp[0].item(), -cp[1].item()), (0, 1))
        energy = (probe_final.abs() ** 2).sum() + 1e-12
        probe_final = probe_final * torch.sqrt(float(Mp * Np) / energy)

    return obj_final.cpu().numpy(), probe_final.cpu().numpy(), loss_hist, model


# =========================================================================
# Evaluation
# =========================================================================

def _normalize(a):
    r = a.max() - a.min()
    return (a - a.min()) / (r + 1e-12)


def ssim_amp_phase(recov_c, gt_c):
    from skimage.metrics import structural_similarity as ssim
    ra, ga = np.abs(recov_c), np.abs(gt_c)
    ssim_a = ssim(_normalize(ga), _normalize(ra), data_range=1.0)
    rp, gp = np.angle(recov_c), np.angle(gt_c)
    piston = np.angle(np.mean(np.exp(1j * (rp - gp))))
    rp_aligned = np.angle(np.exp(1j * (rp - piston)))
    ssim_p = ssim((gp + pi) / (2 * pi),
                  (rp_aligned + pi) / (2 * pi), data_range=1.0)
    return ssim_a, ssim_p, rp_aligned


# =========================================================================
# Panels
# =========================================================================

def save_summary_figure(out_path, *, gt_amp, gt_phase, rec_amp, rec_phase_aligned,
                         ssim_amp, ssim_phase,
                         gt_probe_amp, gt_probe_phase, rec_probe_amp, rec_probe_phase,
                         loss_hist):
    fig, axes = plt.subplots(3, 2, figsize=(11, 15))

    panels = [
        (axes[0, 0], gt_amp, 'GT Object Amplitude', 'gray', 0, gt_amp.max()),
        (axes[0, 1], rec_amp,
         f'Recov Object Amplitude (SSIM={ssim_amp:.3f})', 'gray',
         0, rec_amp.max()),
        (axes[1, 0], gt_phase, 'GT Object Phase', 'inferno', -pi / 2, pi / 2),
        (axes[1, 1], rec_phase_aligned,
         f'Recov Object Phase (SSIM={ssim_phase:.3f})', 'inferno', -pi / 2, pi / 2),
        (axes[2, 0], gt_probe_amp, 'GT Probe Amplitude', 'gray',
         0, gt_probe_amp.max()),
        (axes[2, 1], rec_probe_amp, 'Recov Probe Amplitude (normalized)', 'gray',
         0, rec_probe_amp.max()),
    ]
    for ax, data, title, cmap, vmin, vmax in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


# =========================================================================
# Main
# =========================================================================

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    probe_np = loadmat(PROBE_FILE)['probe']
    probe = torch.from_numpy(probe_np).to(device).to(torch.complex64)
    M, N = probe.shape
    print(f'Probe: {probe.shape}, |probe| in [{np.abs(probe_np).min():.3e}, '
          f'{np.abs(probe_np).max():.3e}]')

    tlX, tlY, brX, brY, canvas, xlocation, ylocation = build_scan_positions((M, N))
    print(f'Scan: {len(tlX)} frames (from plant dataset), canvas={canvas}')
    save_scan_path(tlX, tlY, xlocation, ylocation, (M, N), canvas, RESULT_DIR)
    print(f'  Saved scan path to {RESULT_DIR}/scan_path.png')

    gt_obj, gt_amp, gt_phase = load_gt_object(canvas)
    print(f'GT object: {gt_obj.shape}, amp in [{gt_amp.min():.3f}, {gt_amp.max():.3f}], '
          f'phase in [{gt_phase.min():.2f}, {gt_phase.max():.2f}]')

    print('Simulating measurements...')
    im_raw = simulate(gt_obj, probe, tlX, tlY, brX, brY, device)
    print(f'im_raw: {tuple(im_raw.shape)}, '
          f'[{im_raw.min().item():.3e}, {im_raw.max().item():.3e}]')

    # Save simulated bundle so Recovery.py can be re-run independently
    bundle_path = os.path.join(RESULT_DIR, 'sim_bundle.mat')
    savemat(bundle_path, {
        'xlocation': xlocation[None, :], 'ylocation': ylocation[None, :],
        'probe': probe_np, 'imRaw': im_raw.cpu().numpy(),
        'obj': gt_obj, 'initProbe': probe_np,
    }, do_compression=True)
    print(f'Saved simulated bundle: {bundle_path}')

    print('Reconstructing...')
    obj_rec, probe_rec, loss_hist, model = reconstruct(
        im_raw, (M, N), canvas, tlX, tlY, brX, brY, device)

    ssim_a, ssim_p, phase_aligned = ssim_amp_phase(obj_rec, gt_obj)
    print(f'\nSSIM  object   amp = {ssim_a:.4f}  phase = {ssim_p:.4f}')

    np.save(os.path.join(RESULT_DIR, 'object_recovery.npy'), obj_rec)
    np.save(os.path.join(RESULT_DIR, 'probe_recovery.npy'), probe_rec)
    with open(os.path.join(RESULT_DIR, 'ssim.txt'), 'w', encoding='utf-8') as f:
        f.write(f'ssim_obj_amp\t{ssim_a:.6f}\n')
        f.write(f'ssim_obj_phase\t{ssim_p:.6f}\n')
    with open(os.path.join(RESULT_DIR, 'loss_history.txt'), 'w', encoding='utf-8') as f:
        for i, v in enumerate(loss_hist):
            f.write(f'{i}\t{v:.6f}\n')

    save_summary_figure(
        os.path.join(RESULT_DIR, 'gt_vs_recon.png'),
        gt_amp=gt_amp, gt_phase=gt_phase,
        rec_amp=np.abs(obj_rec), rec_phase_aligned=phase_aligned,
        ssim_amp=ssim_a, ssim_phase=ssim_p,
        gt_probe_amp=np.abs(probe_np), gt_probe_phase=np.angle(probe_np),
        rec_probe_amp=np.abs(probe_rec), rec_probe_phase=np.angle(probe_rec),
        loss_hist=loss_hist,
    )

    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel('epoch')
    plt.ylabel('mean batch loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'loss_curve.png'), dpi=120)
    plt.close()

    save_model_with_required_grad(
        model, os.path.join(RESULT_DIR, 'trained_models.pth'))

    print(f'\nResults saved to: {RESULT_DIR}')
    return ssim_a, ssim_p


if __name__ == '__main__':
    main()
