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

There is **no test suite, no linter, no build step**. `tinycudann` must be installed for the `HashGrid` encoder; `torch`, `scipy`, `pyyaml`, `matplotlib`, `tqdm`, `h5py` cover the rest. Default env is conda `CP_torch` (PyTorch 2.8 + CUDA 12.8).

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

Object canvas size is computed from `max(tlY+M, tlX+N)` rounded up to a multiple of 6 — don't hard-code it.

### Model (`models/complex_inr.py`, `ComplexINRModel2D`)

Two independent branches, each: `tinycudann.Encoding(HashGrid)` → `ComplexMLP` (complex-exponential activation `exp(i·ω·Wx)`, final complex `Linear`):

- **Object branch**: generated at `output_size / downsample_factor`, then Fourier zero-pad upsampled back to full canvas (not bilinear — phase must survive).
- **Probe branch**: generated at the probe's native size (e.g. 512×512).
- **Residual mode** (`use_residual=True`): `output = k * mlp_output + initial_pattern`, where `k` is a learnable scalar initialized to `1e-8`, and `initial_pattern` is a frozen buffer (e.g. a measured probe). This lets the model start from a prior and slowly add corrections.
- `init_type` selects what goes into `probe_initial`: `initProbe` uses the measured probe, `initQuadratic` multiplies it by a Fresnel quadratic phase, `initNone` disables the residual entirely.

The `object_complex_mlp.net(coords)` call path bypasses `ComplexMLP.forward`'s trailing `.real` — the full model needs the complex output. `_get_complex_output` exists for this.

### Training loop (inside `Recovery.py`)

Every batch does, in order:

1. Forward through the model → `(object_complex, probe_complex)`.
2. **L2-normalize the probe**: `probe /= sqrt(sum(|probe|^2))`. This removes the object/probe scale ambiguity; the object branch is free to scale.
3. **Probe center-of-mass roll**: keeps the probe centered so translation symmetry doesn't drift. Computed in float32 (the bfloat16 path underflows and produces NaN shifts).
4. Per-frame loss accumulation (frames in a mini-batch share one forward pass), then `backward(retain_graph=True)` except on the last batch.
5. AMP bfloat16 autocast wraps the whole forward+loss; complex ops stay at complex64 but real reductions/sqrts happen at lower precision.

The loop mirrors `legacy/plant/Recovery.py` line-by-line — if you modify one, mirror changes consciously.

### Data loading

Two accepted layouts, selected by config keys:

- **Bundle** (`bundle_file`): a single `.mat` containing `xlocation`, `ylocation`, `probe`, `imRaw`, optionally `obj` and `initProbe`. Used by `plant`.
- **Split** (`location_file` + `raw_file` + `probe_file` + optional `gt_file` + `init_probe_file`): separate `.mat` files. Used by `clam`, `simulation`.

`utils/io.py:read_data` transparently handles MATLAB v5 (`scipy.io.loadmat`) and v7.3 (HDF5 via `h5py`) including complex-valued arrays stored as compound `{real, imag}` dtype.

### Legacy vs current

`legacy/` is historical/reference code. `models/complex_inr.py` is a verbatim copy of `legacy/network_complex_euler.py` (kept in sync intentionally).

`gscp/` is carried over from the sibling `coded-ptychography-reconstruction` project. Nothing in `Recovery.py` or `models/` references it; leave it alone unless you are actively introducing Gaussian splatting.

## Gotchas

- **Don't push to `main` without authorization.** Remote is `https://github.com/zhixuanh0710/conventional-ptychography-reconstruction.git`.
- **Windows stdout encoding**: `Recovery.py` calls `sys.stdout.reconfigure(encoding='utf-8')` at the top because redirected stdout defaults to cp1252 and fails on non-ASCII output.
- **Not all hyperparameter combos converge.** With `init_type: initNone`, bs=10, lr=1e-3 (the plant default), loss plateaus around 1e3 on plant after 2000 epochs. `initProbe` + `use_residual: true`, or larger batch + higher lr (e.g. bs=25 lr=2e-3), are required to drive loss further down. Plateau is *not* a bug.
- **`torch.roll` shift args**: newer PyTorch rejects CUDA tensor scalars — use `.item()`. The rolled probe centering path has an `isfinite` guard that silently skips when the COM computation produces NaN/Inf under bfloat16.
- **Non-square probes** are not supported. The probe reshape convention `(probe_width, probe_height)` from the legacy code transposes shape on non-square inputs; assert or refuse rather than guessing.
- **`.mat`/`.pth`/`.ipynb`/`*.log` are gitignored** (legacy notebooks are >100MB). If adding new large binaries, update `.gitignore` rather than relying on "I won't add it."
