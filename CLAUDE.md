# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the reconstruction

The entire project is driven by one script and one YAML config:

```bash
python Recovery.py --config config/plant.yaml
python Recovery.py --config config/clam.yaml
python Recovery.py --config config/simulation.yaml
```

Outputs go to `cfg.result_dir` (e.g. `results/plant/`): `e_{epoch}.png`, `result.mat`, `trained_models.pth`, checkpointed every `vis_interval` epochs and at the final epoch. `.mat`/`.pth`/logs/`results/` are gitignored — never commit them.

There is **no test suite, no linter, no build step**. `tinycudann` must be installed for the `HashGrid` encoder; `torch`, `scipy`, `pyyaml`, `matplotlib`, `tqdm`, `h5py` cover the rest. `README.md` has the pip recipe; `AGENTS.md` (sibling contributor guide) largely repeats this doc — keep both in sync if you change conventions.

### Python environment

The required packages live in the user conda env `CP_torch` (PyTorch 2.8 + CUDA 12.8). The shell's default `python` (base conda) does **not** have `tinycudann`. Run scripts with the absolute path to that env's interpreter:

```bash
"C:/Users/Lab/.conda/envs/CP_torch/python.exe" -u Recovery.py --config config/plant.yaml
```

## Architecture

### Forward model (conventional far-field ptychography)

```
exit_wave  = probe * object[tlY:brY, tlX:brX]
measurement = |fftshift(fft2(exit_wave))|^2
loss        = smooth_L1 / FD / Poisson( measurement, captured_frame )
```

Scan positions are physical (meters) and must be projected onto the sample-plane pixel grid before indexing:

```
dx  = wavelength * camera_length / (probe_shape * camera_pixel_pitch)
tlY = round((ylocation - ylocation.min()) / dx[0])
tlX = round((xlocation - xlocation.min()) / dx[1])
```

Object canvas size is computed from `max(tlY+M, tlX+N)` rounded up to a multiple of 6 — don't hard-code it. `utils/optics.py:scan_positions_to_pixels` already returns `(tlX, tlY, brX, brY, dx, canvas_size)`; use it instead of re-deriving.

### Model (`models/complex_inr.py`, `ComplexINRModel2D`)

Two independent branches, each: `tinycudann.Encoding(HashGrid)` → `ComplexMLP` (complex-exponential activation `exp(i·ω·Wx)`, final complex `Linear`):

- **Object branch**: generated at `output_size / downsample_factor`, then Fourier zero-pad upsampled back to full canvas (not bilinear — phase must survive).
- **Probe branch**: generated at the probe's native size (e.g. 512×512).
- **Residual mode** (`use_residual=True`): `output = k * mlp_output + initial_pattern`, where `k` is a learnable scalar initialized to `1e-8`, and `initial_pattern` is a frozen buffer (e.g. a measured probe). This lets the model start from a prior and slowly add corrections.
- `init_type` selects what goes into `probe_initial`: `initProbe` uses the measured probe, `initQuadratic` multiplies it by a Fresnel quadratic phase, `initNone` disables the residual entirely.

The `object_complex_mlp.net(coords)` call path bypasses `ComplexMLP.forward`'s trailing `.real` — the full model needs the complex output. `_get_complex_output` exists for this.

### Alternate model: Gaussian splatting (`models/gaussian_fields.py`)

`ConventionalGSModel2D` (plus its `ObjectGaussianField2D` / `ProbeGaussianField2D` components) is a drop-in alternative to `ComplexINRModel2D` that wraps `gscp.models.GaussianFieldModel`. Both return `(object_complex, probe_complex)` from a no-arg `forward()`.

It is selected by `model_type: gaussian_field` in YAML (default is `inr`). When chosen, `Recovery.py`:

- Reads two nested config dicts — `gaussian_field` (object branch) and `probe_gaussian_field` (probe branch) — for densification schedule, parameterization, MCMC knobs, etc.
- Builds the optimizer from `object_model.get_param_groups(base_lr)` and the probe's groups (probe groups get a `probe_` name prefix so the SGLD step can pick them out by group name).
- After each `loss.backward()`, calls `accumulate_gradients()` on both fields, then `optimizer.step()`, then `densification_step(optimizer)` on each. A plain Adam step does not densify on its own.
- Under `density_control: mcmc`, additionally calls `field.sgld_noise_step(lr_xy, lr_scale)` for SGLD noise injection on xy + scale params; the call no-ops for `density_control: adc`.

### Training loop (inside `Recovery.py`)

Every batch does, in order:

1. Forward through the model → `(object_complex, probe_complex)`.
2. **Probe center-of-mass roll**: keeps the probe centered so translation symmetry doesn't drift. See the Gotcha below on centroid math under autocast.
3. **L2-normalize the probe** (`probe /= sqrt(sum(|probe|^2))`) to break the object/probe scale ambiguity; the object branch is free to scale.
4. Per-frame loss accumulation (frames in a mini-batch share one forward pass), then `backward(retain_graph=True)` except on the last batch.
5. AMP bfloat16 autocast wraps the whole forward+loss; complex ops stay at complex64 but real reductions/sqrts happen at lower precision.

The loop mirrors `legacy/plant/Recovery.py` line-by-line — if you modify one, mirror changes consciously.

### Data loading

Two accepted layouts, selected by config keys:

- **Bundle** (`bundle_file`): a single `.mat` containing `xlocation`, `ylocation`, `probe`, `imRaw`, optionally `obj` and `initProbe`. Used by `plant`.
- **Split** (`location_file` + `raw_file` + `probe_file` + optional `gt_file` + `init_probe_file`): separate `.mat` files. Used by `clam`, `simulation`.

`utils/io.py:load_ptychography_data` dispatches both; `read_data` transparently handles MATLAB v5 (`scipy.io.loadmat`) and v7.3 (HDF5 via `h5py`) including complex-valued arrays stored as compound `{real, imag}` dtype.

### Simulation experiments and analysis scripts

- `scripts/simulate_and_reconstruct.py`: generates synthetic measurements from a GT image (amplitude = grayscale PNG, phase = `(1-amp)*π - π/2`) and a real-probe `.mat`, runs the same reconstruction loop inline, reports SSIM on object amp/phase. Writes a `sim_bundle.mat` compatible with Recovery.py so the same data can be re-run via a config. Reference implementation: `h:\My Drive\coded-ptychography-reconstruction\scripts\simulate_and_reconstruct.py`.
- `scripts/compare_object_ssim.py`: compares `result.mat` to GT object. Handles three ambiguities: (1) recon-canvas-smaller-than-GT (max-cross-correlation alignment + crop), (2) global-phase ambiguity (estimates `exp(i·alpha)`), (3) probe/object scale ambiguity (rescales amplitudes to mean=1). Phase SSIM is on wrapped phase mapped to `[0,1]` via `(phi+pi)/(2pi)`. The `--coverage-frac` flag restricts the SSIM mask to pixels covered by ≥frac·max scan visits.
- `scripts/compare_ssim_plot.py`: 2×4 GT-vs-recon panel (obj amp/phase, probe amp/phase) with per-column SSIM annotations.
- `scripts/sweep_object_gaussians.py`: sweeps object-branch `num_initial_gaussians` (= `max_gaussians`, pinning the count) by cloning a base GS YAML, redirecting `result_dir` to `results/sweep_ng{N}/`, running Recovery.py, and parsing SSIM via `compare_object_ssim.py`. Hard-codes the `CP_torch` python at the top of the file.
- `scripts/save_sweep_figs.py`: walks every `results/sweep_ng<N>/result.mat`, saves per-panel PNGs (obj amp gray, obj phase inferno, probe amp gray, probe phase viridis) and a combined SSIM CSV. Alignment/scale/phase handling mirrors `compare_object_ssim.py`.

### Legacy vs current

`legacy/` is historical/reference code. `models/complex_inr.py` is a verbatim copy of `legacy/network_complex_euler.py` (kept in sync intentionally).

`gscp/` is vendored from the sibling `coded-ptychography-reconstruction` project. It is now an active transitive dependency: `Recovery.py` → `models.gaussian_fields` → `gscp.models.GaussianFieldModel`. Do not delete or "tidy" it. The CUDA extension under `gscp/csrc/` is optional — the GS path falls back to a pure-PyTorch rasterizer if the kernel isn't built.

A separate reference implementation of this forward model lives at `h:\My Drive\gscp\gscp\training\conventional_trainer.py`. Consult it when debugging probe centering / normalization — that version reliably produces centered probes.

### Config schema

`Recovery.py`'s `load_config` does **one level** of flattening: every top-level dict-valued section (`data`, `physics`, `model`, `training`, `output`, …) gets merged into one flat cfg dict. Nested dicts inside those sections (e.g. `model.gaussian_field`, `model.probe_gaussian_field`) stay nested — the GS path reads them via `cfg.get('gaussian_field', {})` etc.

So both schema styles work as long as the leaf key names match what Recovery.py expects (`iters`, `lr`, `batch_size`, `init_type`, `result_dir`, `model_type`, …). The flat plant/clam/simulation configs and the section-grouped GS configs (`sim_ng1000_reloc.yaml`, `simulation_conventional_westwest_ng50000.yaml`) all load the same way.

The trap is leaf-name drift: `iterations` instead of `iters`, or `learning_rate` instead of `lr`, will silently fall back to the hard-coded defaults inside `Recovery.py`. Match the names used by the working configs.

## Gotchas

- **Don't push to `main` without authorization.** Remote is `https://github.com/zhixuanh0710/conventional-ptychography-reconstruction.git`.
- **Windows stdout encoding**: `Recovery.py` calls `sys.stdout.reconfigure(encoding='utf-8')` at the top because redirected stdout defaults to cp1252 and fails on non-ASCII output. Any new script that prints non-ASCII must do the same.
- **Probe centering must run in `torch.no_grad()` outside autocast**. The COM math uses `cumsum(0).mean()` on the probe marginals; under bfloat16 autocast the accumulator loses enough precision that `cp = trunc(...).long()` overflows or clamps to junk, leaving the probe stuck at a canvas corner. The working pattern (see `gscp`'s conventional trainer) casts `absP2` to fp32 inside a `no_grad` block, computes `cp`, and rolls the complex tensor outside the block.
- **Probe energy normalization target matters for bf16 stability**. Normalizing to `sum(|probe|²) = 1` pushes per-pixel amplitude to `~1/sqrt(N_pixels) ≈ 2e-3`, which underflows in bfloat16. Normalizing to `sum(|probe|²) = N_pixels` keeps per-pixel amplitude on order 1. Recovery.py's legacy-mirror uses L2=1 and works because the float32 cast in the COM step is enough; simulation scripts that saw garbage centered probes were using L2=1 without the fp32 cast.
- **Conventional ptychography has a probe/object scale coupling**. The measurement `|FFT(probe·obj_patch)|²` is invariant to `probe *= k, obj /= k`, so absolute amplitude scale is arbitrary. When plotting GT vs recovery, use per-panel `vmax` (not a shared one) or SSIM on normalized amplitudes — a faithful reconstruction can look "all black" under a GT-scaled colorbar.
- **Not all hyperparameter combos converge.** With `init_type: initNone`, bs=10, lr=1e-3 (the plant default), loss plateaus around 1e3 on plant after 2000 epochs. `initProbe` + `use_residual: true`, or larger batch + higher lr (e.g. bs=25 lr=2e-3), are required to drive loss further down. Plateau is *not* a bug.
- **`torch.roll` shift args**: newer PyTorch rejects CUDA tensor scalars — use `.item()`. The rolled probe centering path has an `isfinite` guard that silently skips when the COM computation produces NaN/Inf under bfloat16.
- **Non-square probes** are not supported. The probe reshape convention `(probe_width, probe_height)` from the legacy code transposes shape on non-square inputs; assert or refuse rather than guessing.
- **`.mat`/`.pth`/`.ipynb`/`*.log` are gitignored** (legacy notebooks are >100MB). If adding new large binaries, update `.gitignore` rather than relying on "I won't add it."
