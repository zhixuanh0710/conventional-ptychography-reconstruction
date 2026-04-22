"""Compute SSIM between the reconstructed object (result.mat) and the GT.

Ptychography has a probe/object scale ambiguity and a global-phase ambiguity,
and in this project the recon canvas (sized from scan positions) is smaller
than the GT canvas. We handle all three:

1. Locate the recon in the GT by maximum-cross-correlation of magnitudes.
2. Crop the GT to the matched sub-region.
3. Remove the global phase (estimate exp(i·alpha) that aligns recon to GT).
4. Rescale amplitudes to mean=1 before computing SSIM.

Phase SSIM is computed on the wrapped phase mapped to [0, 1] via (phi+pi)/(2pi).
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim


def _read_complex(d: dict, *keys: str) -> np.ndarray:
    for k in keys:
        if k in d:
            arr = np.asarray(d[k]).squeeze()
            if np.iscomplexobj(arr):
                return arr
            return arr.astype(np.complex64)
    raise KeyError(f"none of {keys!r} found in {list(d.keys())!r}")


def scan_coverage_mask(xlocation, ylocation, probe_shape, canvas_shape,
                       wavelength, camera_length, camera_pixel_pitch):
    """Count how many probe footprints cover each canvas pixel.

    Mirrors utils.optics.scan_positions_to_pixels: min-shifted scan positions,
    projected to pixel grid using dx = λ·L / (M·pitch)."""
    M, N = probe_shape
    Hc, Wc = canvas_shape
    x_pos = np.asarray(xlocation).squeeze() - np.min(xlocation)
    y_pos = np.asarray(ylocation).squeeze() - np.min(ylocation)
    dx = wavelength * camera_length / (np.array([M, N]) * camera_pixel_pitch)
    tlY = np.round(y_pos / dx[0]).astype(int)
    tlX = np.round(x_pos / dx[1]).astype(int)
    cov = np.zeros((Hc, Wc), dtype=np.int32)
    for y, x in zip(tlY, tlX):
        y0, y1 = max(0, y), min(Hc, y + M)
        x0, x1 = max(0, x), min(Wc, x + N)
        if y1 > y0 and x1 > x0:
            cov[y0:y1, x0:x1] += 1
    return cov


def high_coverage_bbox(cov: np.ndarray, frac: float = 0.5) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) — bbox of pixels covered by >= frac * max(cov)."""
    thr = max(1, int(math.ceil(frac * cov.max())))
    mask = cov >= thr
    ys, xs = np.where(mask)
    if ys.size == 0:
        H, W = cov.shape
        return 0, H, 0, W
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def align_by_xcorr(recon_mag: np.ndarray, gt_mag: np.ndarray) -> tuple[int, int]:
    """Return (row_offset, col_offset) s.t. gt_mag[row:row+H, col:col+W] aligns with recon_mag."""
    H, W = recon_mag.shape
    Hg, Wg = gt_mag.shape
    if Hg < H or Wg < W:
        raise ValueError(f"GT ({Hg}x{Wg}) smaller than recon ({H}x{W})")

    # Zero-pad recon to GT size, cross-correlate via FFT, report peak.
    rf = np.fft.fft2(recon_mag - recon_mag.mean(), s=(Hg, Wg))
    gf = np.fft.fft2(gt_mag - gt_mag.mean(), s=(Hg, Wg))
    xc = np.fft.ifft2(gf * np.conj(rf)).real
    # Only consider offsets that keep the recon-sized window inside the GT.
    valid = xc[:Hg - H + 1, :Wg - W + 1]
    idx = np.unravel_index(int(np.argmax(valid)), valid.shape)
    return int(idx[0]), int(idx[1])


def resolve_global_phase(recon: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return recon * exp(i*alpha) where alpha minimises ||recon*exp(i*alpha) - gt||."""
    inner = np.vdot(recon.ravel(), gt.ravel())   # sum(conj(recon) * gt)
    if abs(inner) < 1e-20:
        return recon
    phase = inner / abs(inner)
    return recon * phase


def normalize_amp(x: np.ndarray) -> np.ndarray:
    m = np.abs(x).mean()
    return x / m if m > 0 else x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--recon', default='results/simulation_conventional_westwest_ng50000/result.mat')
    ap.add_argument('--gt',    default='data/simulation_conventional_westwest/dataset-sim.mat')
    ap.add_argument('--wavelength',        type=float, default=6.75e-7)
    ap.add_argument('--camera-length',     type=float, default=0.0295)
    ap.add_argument('--camera-pixel-pitch', type=float, default=2.96e-5)
    ap.add_argument('--coverage-frac', type=float, default=0.5,
                    help="Keep pixels covered by >= this fraction of max coverage "
                         "for the scan-coverage SSIM (0 disables coverage masking).")
    args = ap.parse_args()

    recon_path = Path(args.recon)
    gt_path = Path(args.gt)

    gt_d = loadmat(str(gt_path))
    recon = _read_complex(loadmat(str(recon_path)), 'object_complex', 'object')
    gt    = _read_complex(gt_d,    'gt_object', 'obj', 'object')

    print(f"recon: {recon.shape} {recon.dtype}   (from {recon_path})")
    print(f"gt   : {gt.shape} {gt.dtype}   (from {gt_path})")

    # 1. Align recon in GT by magnitude cross-correlation.
    r0, c0 = align_by_xcorr(np.abs(recon), np.abs(gt))
    H, W = recon.shape
    gt_crop = gt[r0:r0 + H, c0:c0 + W]
    print(f"best crop offset (row, col) = ({r0}, {c0})")

    # 2. Resolve global phase.
    recon_aligned = resolve_global_phase(recon, gt_crop)

    # 3. Amplitude SSIM: normalize both to mean=1, use shared data_range.
    a_r = np.abs(normalize_amp(recon_aligned)).astype(np.float32)
    a_g = np.abs(normalize_amp(gt_crop)).astype(np.float32)
    dr = float(max(a_r.max(), a_g.max()) - min(a_r.min(), a_g.min()))
    ssim_amp = ssim(a_r, a_g, data_range=dr)

    # 4. Phase SSIM: wrap to [-pi, pi], map to [0, 1].
    p_r = (np.angle(recon_aligned).astype(np.float32) + np.pi) / (2 * np.pi)
    p_g = (np.angle(gt_crop).astype(np.float32)       + np.pi) / (2 * np.pi)
    ssim_phase = ssim(p_r, p_g, data_range=1.0)

    # Amplitude-weighted phase SSIM (ignore regions where GT amplitude is tiny).
    mask = a_g > 0.15 * a_g.max()
    if mask.sum() > 100:
        # Localised SSIM on the mask bbox for robustness.
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        ssim_amp_m = ssim(a_r[y0:y1, x0:x1], a_g[y0:y1, x0:x1],
                          data_range=float(max(a_r[y0:y1, x0:x1].max(),
                                               a_g[y0:y1, x0:x1].max())))
        ssim_phase_m = ssim(p_r[y0:y1, x0:x1], p_g[y0:y1, x0:x1], data_range=1.0)
    else:
        ssim_amp_m = float('nan')
        ssim_phase_m = float('nan')

    print()
    print(f"SSIM (amplitude, full canvas)  : {ssim_amp:.4f}")
    print(f"SSIM (phase,     full canvas)  : {ssim_phase:.4f}")
    print(f"SSIM (amplitude, amp-mask bbox): {ssim_amp_m:.4f}")
    print(f"SSIM (phase,     amp-mask bbox): {ssim_phase_m:.4f}")

    # --- Scan-coverage-masked SSIM -----------------------------------------
    if args.coverage_frac > 0 and {'xlocation', 'ylocation', 'probe'}.issubset(gt_d.keys()):
        probe_shape = tuple(np.asarray(gt_d['probe']).squeeze().shape)
        cov = scan_coverage_mask(
            gt_d['xlocation'], gt_d['ylocation'], probe_shape,
            recon.shape, args.wavelength, args.camera_length, args.camera_pixel_pitch,
        )
        y0, y1, x0, x1 = high_coverage_bbox(cov, frac=args.coverage_frac)
        cov_pct_kept = 100 * (y1 - y0) * (x1 - x0) / cov.size
        a_r_c = a_r[y0:y1, x0:x1]
        a_g_c = a_g[y0:y1, x0:x1]
        p_r_c = p_r[y0:y1, x0:x1]
        p_g_c = p_g[y0:y1, x0:x1]
        ssim_amp_c = ssim(a_r_c, a_g_c,
                          data_range=float(max(a_r_c.max(), a_g_c.max())
                                           - min(a_r_c.min(), a_g_c.min())))
        ssim_phase_c = ssim(p_r_c, p_g_c, data_range=1.0)
        print(
            f"SSIM (amplitude, cov>={args.coverage_frac:.2f}·max, "
            f"bbox {x1-x0}x{y1-y0} = {cov_pct_kept:.1f}% canvas): {ssim_amp_c:.4f}"
        )
        print(
            f"SSIM (phase,     cov>={args.coverage_frac:.2f}·max, "
            f"bbox {x1-x0}x{y1-y0} = {cov_pct_kept:.1f}% canvas): {ssim_phase_c:.4f}"
        )
        print(
            f"  coverage: min={cov.min()}, max={cov.max()}, "
            f"threshold={math.ceil(args.coverage_frac * cov.max())}"
        )


if __name__ == '__main__':
    main()
