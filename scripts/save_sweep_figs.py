"""For every results/sweep_ng<N>/result.mat:
    - save obj amp (gray), obj phase (inferno), probe amp (gray), probe phase (viridis)
    - compute SSIM vs GT (object uses scan-coverage mask @ --coverage-frac; probe is full)
Write a combined CSV of SSIMs across all ng values.

Alignment and scale/phase handling mirror scripts/compare_object_ssim.py.
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim


# --- Shared helpers (duplicated from compare_object_ssim.py for self-containment) ---

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
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(valid)), valid.shape))


def resolve_global_phase(r: np.ndarray, g: np.ndarray) -> np.ndarray:
    inner = np.vdot(r.ravel(), g.ravel())
    return r if abs(inner) < 1e-20 else r * (inner / abs(inner))


def normalize_amp(x: np.ndarray) -> np.ndarray:
    m = np.abs(x).mean()
    return x / m if m > 0 else x


def center_of_mass_shift(x: np.ndarray) -> tuple[int, int]:
    absP2 = np.abs(x) ** 2
    H, W = absP2.shape
    tot = absP2.sum()
    if tot == 0:
        return 0, 0
    ys = np.arange(H); xs = np.arange(W)
    cy = (absP2.sum(1) * ys).sum() / tot
    cx = (absP2.sum(0) * xs).sum() / tot
    return int(round(H / 2 - cy)), int(round(W / 2 - cx))


def scan_coverage_mask(xloc, yloc, probe_shape, canvas_shape,
                       wavelength, camera_length, camera_pixel_pitch):
    M, N = probe_shape
    Hc, Wc = canvas_shape
    x_pos = np.asarray(xloc).squeeze() - np.min(xloc)
    y_pos = np.asarray(yloc).squeeze() - np.min(yloc)
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


def high_coverage_bbox(cov: np.ndarray, frac: float) -> tuple[int, int, int, int]:
    thr = max(1, int(math.ceil(frac * cov.max())))
    mask = cov >= thr
    ys, xs = np.where(mask)
    if ys.size == 0:
        H, W = cov.shape
        return 0, H, 0, W
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


# --- SSIM ---

def ssim_amp_phase(r: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    a_r = np.abs(normalize_amp(r)).astype(np.float32)
    a_g = np.abs(normalize_amp(g)).astype(np.float32)
    dr = float(max(a_r.max(), a_g.max()) - min(a_r.min(), a_g.min()))
    s_amp = ssim(a_r, a_g, data_range=dr)
    p_r = (np.angle(r).astype(np.float32) + np.pi) / (2 * np.pi)
    p_g = (np.angle(g).astype(np.float32) + np.pi) / (2 * np.pi)
    s_ph = ssim(p_r, p_g, data_range=1.0)
    return float(s_amp), float(s_ph)


# --- Saving ---

def save_panel(arr: np.ndarray, path: Path, cmap: str,
               vlim: tuple[float, float] | None = None, **_ignored) -> None:
    """Write a raw colormapped PNG — no axes, colorbar, title, or padding.

    `plt.imsave` emits a file whose pixel dimensions equal arr.shape exactly,
    so there is no whitespace to trim.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {'cmap': cmap}
    if vlim is not None:
        kw['vmin'], kw['vmax'] = vlim
    plt.imsave(str(path), arr, **kw)


def parse_ng(name: str) -> int | None:
    m = re.fullmatch(r"sweep_ng(\d+)", name)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--gt', default='data/simulation_conventional_westwest/dataset-sim.mat')
    ap.add_argument('--wavelength',        type=float, default=6.75e-7)
    ap.add_argument('--camera-length',     type=float, default=0.0295)
    ap.add_argument('--camera-pixel-pitch', type=float, default=2.96e-5)
    ap.add_argument('--coverage-frac', type=float, default=0.5)
    ap.add_argument('--out-csv', default='results/sweep_ssim_summary.csv')
    args = ap.parse_args()

    root = Path(args.results_root)
    gt_d = loadmat(args.gt)
    obj_g_full   = _read_complex(gt_d, 'gt_object', 'obj', 'object')
    probe_g_full = _read_complex(gt_d, 'probe', 'initProbe')

    dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and parse_ng(d.name) is not None],
        key=lambda d: parse_ng(d.name),
    )

    rows = []
    for d in dirs:
        ng = parse_ng(d.name)
        result_mat = d / 'result.mat'
        if not result_mat.exists():
            print(f"[skip] {d.name}: no result.mat")
            continue

        rec = loadmat(str(result_mat))
        obj_r   = _read_complex(rec, 'object_complex', 'object')
        probe_r = _read_complex(rec, 'probe_complex',  'probe')

        # --- Object: xcorr-align recon to GT, resolve phase ---
        r0, c0 = align_by_xcorr(np.abs(obj_r), np.abs(obj_g_full))
        H, W = obj_r.shape
        obj_g_crop = obj_g_full[r0:r0 + H, c0:c0 + W]
        obj_r_aligned = resolve_global_phase(obj_r, obj_g_crop)

        # --- Probe: center each, then resolve phase ---
        sy, sx = center_of_mass_shift(probe_r);  probe_r_c = np.roll(probe_r, (sy, sx), axis=(0, 1))
        sy, sx = center_of_mass_shift(probe_g_full);  probe_g_c = np.roll(probe_g_full, (sy, sx), axis=(0, 1))
        probe_r_aligned = resolve_global_phase(probe_r_c, probe_g_c)

        # --- Object SSIM under coverage mask ---
        if {'xlocation', 'ylocation', 'probe'}.issubset(gt_d.keys()):
            probe_shape = tuple(np.asarray(gt_d['probe']).squeeze().shape)
            cov = scan_coverage_mask(
                gt_d['xlocation'], gt_d['ylocation'], probe_shape, obj_r.shape,
                args.wavelength, args.camera_length, args.camera_pixel_pitch,
            )
            y0, y1, x0, x1 = high_coverage_bbox(cov, frac=args.coverage_frac)
            s_obj_amp, s_obj_phase = ssim_amp_phase(
                obj_r_aligned[y0:y1, x0:x1], obj_g_crop[y0:y1, x0:x1],
            )
            bbox_tag = f"cov>={args.coverage_frac:.2f}·max, {x1-x0}x{y1-y0}"
        else:
            s_obj_amp, s_obj_phase = ssim_amp_phase(obj_r_aligned, obj_g_crop)
            bbox_tag = "full canvas"

        # --- Probe SSIM: full 512x512 (no coverage mask applies) ---
        s_pr_amp, s_pr_phase = ssim_amp_phase(probe_r_aligned, probe_g_c)

        # --- Save four PNGs per run ---
        obj_amp   = np.abs(normalize_amp(obj_r_aligned)).astype(np.float32)
        obj_phase = np.angle(obj_r_aligned).astype(np.float32)
        pr_amp    = np.abs(normalize_amp(probe_r_aligned)).astype(np.float32)
        pr_phase  = np.angle(probe_r_aligned).astype(np.float32)

        tag = f"ng={ng}"
        save_panel(obj_amp,   d / 'obj_amp.png',   cmap='gray',
                   title=f"Object Amp — {tag}  (SSIM {s_obj_amp:.4f}, {bbox_tag})")
        save_panel(obj_phase, d / 'obj_phase.png', cmap='inferno',
                   title=f"Object Phase — {tag}  (SSIM {s_obj_phase:.4f}, {bbox_tag})",
                   vlim=(-np.pi, np.pi))
        save_panel(pr_amp,    d / 'probe_amp.png', cmap='gray',
                   title=f"Probe Amp — {tag}  (SSIM {s_pr_amp:.4f})")
        save_panel(pr_phase,  d / 'probe_phase.png', cmap='viridis',
                   title=f"Probe Phase — {tag}  (SSIM {s_pr_phase:.4f})",
                   vlim=(-np.pi, np.pi))

        rows.append({
            'ng': ng,
            'obj_amp_ssim':   round(s_obj_amp,   4),
            'obj_phase_ssim': round(s_obj_phase, 4),
            'probe_amp_ssim':   round(s_pr_amp,   4),
            'probe_phase_ssim': round(s_pr_phase, 4),
            'coverage_frac': args.coverage_frac,
        })
        print(f"{d.name}: obj amp={s_obj_amp:.4f} phase={s_obj_phase:.4f}   "
              f"probe amp={s_pr_amp:.4f} phase={s_pr_phase:.4f}   [{bbox_tag}]")

    # --- CSV ---
    if rows:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\ncsv: {out_csv}")

        print("\n==== summary ====")
        print(f"{'ng':>8}  {'obj_amp':>7}  {'obj_phase':>9}  {'pr_amp':>7}  {'pr_phase':>8}")
        for r in rows:
            print(f"{r['ng']:>8}  {r['obj_amp_ssim']:>7.4f}  {r['obj_phase_ssim']:>9.4f}"
                  f"  {r['probe_amp_ssim']:>7.4f}  {r['probe_phase_ssim']:>8.4f}")


if __name__ == '__main__':
    main()
