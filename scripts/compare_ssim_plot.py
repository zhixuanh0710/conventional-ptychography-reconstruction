"""2x4 GT-vs-recon comparison with SSIM annotations.

Layout:
    row 0 = GT:    [obj amp | obj phase | probe amp | probe phase]
    row 1 = recon: [obj amp | obj phase | probe amp | probe phase]

Each column's SSIM is drawn above the recon panel.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim


def _read_complex(d: dict, *keys: str) -> np.ndarray:
    for k in keys:
        if k in d:
            arr = np.asarray(d[k]).squeeze()
            return arr if np.iscomplexobj(arr) else arr.astype(np.complex64)
    raise KeyError(f"none of {keys!r} in {list(d.keys())!r}")


def align_by_xcorr(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    H, W = a.shape
    Hg, Wg = b.shape
    rf = np.fft.fft2(a - a.mean(), s=(Hg, Wg))
    gf = np.fft.fft2(b - b.mean(), s=(Hg, Wg))
    xc = np.fft.ifft2(gf * np.conj(rf)).real
    valid = xc[:Hg - H + 1, :Wg - W + 1]
    idx = np.unravel_index(int(np.argmax(valid)), valid.shape)
    return int(idx[0]), int(idx[1])


def resolve_global_phase(r: np.ndarray, g: np.ndarray) -> np.ndarray:
    inner = np.vdot(r.ravel(), g.ravel())
    if abs(inner) < 1e-20:
        return r
    return r * (inner / abs(inner))


def normalize_amp(x: np.ndarray) -> np.ndarray:
    m = np.abs(x).mean()
    return x / m if m > 0 else x


def center_of_mass_shift(x: np.ndarray) -> tuple[int, int]:
    """Return integer shift that puts centroid of |x|^2 at the array center."""
    absP2 = np.abs(x) ** 2
    H, W = absP2.shape
    tot = absP2.sum()
    if tot == 0:
        return 0, 0
    ys = np.arange(H); xs = np.arange(W)
    cy = (absP2.sum(1) * ys).sum() / tot
    cx = (absP2.sum(0) * xs).sum() / tot
    return int(round(H / 2 - cy)), int(round(W / 2 - cx))


def compare_panel(recon: np.ndarray, gt: np.ndarray,
                  *, center_probe: bool = False) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (recon_aligned, gt_crop, ssim_amp, ssim_phase)."""
    if recon.shape != gt.shape:
        r0, c0 = align_by_xcorr(np.abs(recon), np.abs(gt))
        gt = gt[r0:r0 + recon.shape[0], c0:c0 + recon.shape[1]]

    if center_probe:
        # Roll each one to put its own centroid at array center so the
        # probe panels aren't dominated by a translation offset that
        # doesn't affect reconstruction quality.
        sy, sx = center_of_mass_shift(recon)
        recon = np.roll(recon, (sy, sx), axis=(0, 1))
        sy, sx = center_of_mass_shift(gt)
        gt = np.roll(gt, (sy, sx), axis=(0, 1))

    recon = resolve_global_phase(recon, gt)

    a_r = np.abs(normalize_amp(recon)).astype(np.float32)
    a_g = np.abs(normalize_amp(gt)).astype(np.float32)
    dr = float(max(a_r.max(), a_g.max()) - min(a_r.min(), a_g.min()))
    s_amp = ssim(a_r, a_g, data_range=dr)

    p_r = (np.angle(recon).astype(np.float32) + np.pi) / (2 * np.pi)
    p_g = (np.angle(gt   ).astype(np.float32) + np.pi) / (2 * np.pi)
    s_phase = ssim(p_r, p_g, data_range=1.0)

    return recon, gt, s_amp, s_phase


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--recon', default='results/simulation_conventional_westwest_ng50000/result.mat')
    ap.add_argument('--gt',    default='data/simulation_conventional_westwest/dataset-sim.mat')
    ap.add_argument('--out',   default='results/simulation_conventional_westwest_ng50000/compare_ssim.png')
    args = ap.parse_args()

    recon_d = loadmat(args.recon)
    gt_d    = loadmat(args.gt)

    obj_r   = _read_complex(recon_d, 'object_complex', 'object')
    probe_r = _read_complex(recon_d, 'probe_complex', 'probe')
    obj_g   = _read_complex(gt_d, 'gt_object', 'obj', 'object')
    probe_g = _read_complex(gt_d, 'probe', 'initProbe')

    obj_r_a,   obj_g_a,   s_obj_amp,   s_obj_phase   = compare_panel(obj_r,   obj_g)
    pr_r_a,    pr_g_a,    s_pr_amp,    s_pr_phase    = compare_panel(probe_r, probe_g, center_probe=True)

    print(f"object SSIM:  amp={s_obj_amp:.4f}   phase={s_obj_phase:.4f}")
    print(f"probe  SSIM:  amp={s_pr_amp:.4f}   phase={s_pr_phase:.4f}")

    # Build the 2x4 figure.
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=130)

    def show(ax, img, title, cmap='gray', vlim=None):
        if vlim is None:
            im = ax.imshow(img, cmap=cmap)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 0 — GT
    show(axes[0, 0], np.abs(normalize_amp(obj_g_a)),       'GT — Object Amp')
    show(axes[0, 1], np.angle(obj_g_a),                    'GT — Object Phase', vlim=(-np.pi, np.pi))
    show(axes[0, 2], np.abs(normalize_amp(pr_g_a)),        'GT — Probe Amp')
    show(axes[0, 3], np.angle(pr_g_a),                     'GT — Probe Phase',  vlim=(-np.pi, np.pi))

    # Row 1 — recon (title carries SSIM)
    show(axes[1, 0], np.abs(normalize_amp(obj_r_a)),       f'Recon — Object Amp\nSSIM = {s_obj_amp:.4f}')
    show(axes[1, 1], np.angle(obj_r_a),                    f'Recon — Object Phase\nSSIM = {s_obj_phase:.4f}',
         vlim=(-np.pi, np.pi))
    show(axes[1, 2], np.abs(normalize_amp(pr_r_a)),        f'Recon — Probe Amp\nSSIM = {s_pr_amp:.4f}')
    show(axes[1, 3], np.angle(pr_r_a),                     f'Recon — Probe Phase\nSSIM = {s_pr_phase:.4f}',
         vlim=(-np.pi, np.pi))

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == '__main__':
    main()
