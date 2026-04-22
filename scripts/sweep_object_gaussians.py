"""Sweep object-branch num_initial_gaussians and report object-SSIM@0.5.

For each N in --ng-values:
  1. Clone the base config; set object num_initial_gaussians = max_gaussians = N
     (pins the object count — no densification growth beyond N).
  2. Redirect output to results/sweep_ng{N}/.
  3. Run Recovery.py.
  4. Run compare_object_ssim.py --coverage-frac 0.5; parse the two SSIM numbers.
Finally print a summary table and write it to <out-csv>.
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse
import csv
import re
import shutil
import subprocess
import time
from pathlib import Path

import yaml


PY = r"C:/Users/Lab/.conda/envs/CP_torch/python.exe"


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('wb') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        return proc.wait()


def parse_ssim(stdout: str, frac: float) -> tuple[float, float] | None:
    # "SSIM (amplitude, cov>=0.50·max, bbox ... ): 0.8676"
    pat_amp = re.compile(
        rf"SSIM \(amplitude, cov>={frac:.2f}.max.*?\): ([0-9.]+)"
    )
    pat_ph = re.compile(
        rf"SSIM \(phase,\s+cov>={frac:.2f}.max.*?\): ([0-9.]+)"
    )
    m_a = pat_amp.search(stdout)
    m_p = pat_ph.search(stdout)
    if not m_a or not m_p:
        return None
    return float(m_a.group(1)), float(m_p.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-config', default='config/simulation_conventional_westwest_ng50000.yaml')
    ap.add_argument('--ng-values', nargs='+', type=int,
                    default=[1000, 5000, 10000, 20000, 30000, 40000])
    ap.add_argument('--coverage-frac', type=float, default=0.5)
    ap.add_argument('--out-csv', default='results/sweep_object_ng.csv')
    ap.add_argument('--out-dir-prefix', default='results/sweep_ng')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    base_path = Path(args.base_config)
    with base_path.open('r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    gt_path = Path(base_cfg['data']['data_dir']) / base_cfg['data']['bundle_file']

    rows = []
    for ng in args.ng_values:
        run_dir = Path(f"{args.out_dir_prefix}{ng}")
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy via yaml roundtrip
        cfg['model']['gaussian_field']['num_initial_gaussians'] = int(ng)
        cfg['model']['gaussian_field']['max_gaussians'] = int(ng)
        cfg['output']['result_dir'] = str(run_dir).replace('\\', '/')

        cfg_path = run_dir / 'config.yaml'
        run_dir.mkdir(parents=True, exist_ok=True)
        with cfg_path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        # Clear any stale artefacts from a previous run in this dir.
        for p in run_dir.glob('e_*.png'):
            p.unlink()
        for name in ('result.mat', 'trained_models.pth'):
            p = run_dir / name
            if p.exists():
                p.unlink()

        print(f"\n=== ng={ng} ===  config={cfg_path}  out={run_dir}")
        if args.dry_run:
            rows.append({'ng': ng, 'train_s': 0, 'ssim_s': 0,
                         'ssim_amp': float('nan'), 'ssim_phase': float('nan'),
                         'status': 'dry-run'})
            continue

        # --- Train ---
        t0 = time.time()
        rc = run([PY, '-u', 'Recovery.py', '--config', str(cfg_path)],
                 run_dir / 'train.log')
        t_train = time.time() - t0
        if rc != 0:
            print(f"  TRAIN FAILED (rc={rc}); see {run_dir/'train.log'}")
            rows.append({'ng': ng, 'train_s': t_train, 'ssim_s': 0,
                         'ssim_amp': float('nan'), 'ssim_phase': float('nan'),
                         'status': f'train_rc={rc}'})
            continue
        print(f"  train: {t_train:.1f}s  (exit 0)")

        # --- SSIM ---
        t0 = time.time()
        ssim_log = run_dir / 'ssim.log'
        rc = run(
            [PY, '-u', 'scripts/compare_object_ssim.py',
             '--recon', str(run_dir / 'result.mat'),
             '--gt', str(gt_path),
             '--coverage-frac', f"{args.coverage_frac}"],
            ssim_log,
        )
        t_ssim = time.time() - t0
        text = ssim_log.read_text(encoding='utf-8', errors='replace')
        parsed = parse_ssim(text, args.coverage_frac)
        if rc != 0 or parsed is None:
            print(f"  SSIM PARSE FAILED (rc={rc}); see {ssim_log}")
            rows.append({'ng': ng, 'train_s': t_train, 'ssim_s': t_ssim,
                         'ssim_amp': float('nan'), 'ssim_phase': float('nan'),
                         'status': 'ssim_failed'})
            continue
        amp, ph = parsed
        print(f"  SSIM (cov>={args.coverage_frac:.2f}·max): amp={amp:.4f}  phase={ph:.4f}")
        rows.append({'ng': ng, 'train_s': t_train, 'ssim_s': t_ssim,
                     'ssim_amp': amp, 'ssim_phase': ph, 'status': 'ok'})

    # --- Summary ---
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n==== summary (coverage_frac={:.2f}) ====".format(args.coverage_frac))
    print(f"{'ng':>8}  {'amp':>6}  {'phase':>6}  {'train_s':>8}  status")
    for r in rows:
        print(f"{r['ng']:>8}  {r['ssim_amp']:>6.4f}  {r['ssim_phase']:>6.4f}  {r['train_s']:>8.1f}  {r['status']}")
    print(f"\ncsv: {out_csv}")


if __name__ == '__main__':
    main()
