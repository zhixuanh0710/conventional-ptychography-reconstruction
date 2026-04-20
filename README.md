# Conventional Ptychography Reconstruction

Far-field ptychographic reconstruction driven by a complex-valued implicit neural representation (INR). The object and probe are each parameterized by a `HashGrid` encoder (tiny-cuda-nn) followed by a complex-exponential MLP; the forward model is

```
exit_wave   = probe * object[tlY:brY, tlX:brX]
measurement = |fftshift(fft2(exit_wave))|²
```

and is supervised against the captured diffraction stack via `smooth_L1`, `FD`, or `Poisson` loss.

## Quick start

```bash
# env (conda)
conda create -n CP_torch python=3.10
conda activate CP_torch
pip install torch torchvision scipy pyyaml tqdm matplotlib h5py
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# run
python Recovery.py --config config/plant.yaml
python Recovery.py --config config/clam.yaml
python Recovery.py --config config/simulation.yaml
```

Outputs go to `cfg.output.result_dir` (e.g. `results/plant/`): `e_{epoch}.png` visualizations, `result.mat` (object + probe complex arrays), and `trained_models.pth`.

## Data layout

Two arrangements are supported, selected by the YAML config:

| Config key            | Layout                                                                 |
|-----------------------|------------------------------------------------------------------------|
| `bundle_file`         | A single `.mat` holding `xlocation`, `ylocation`, `probe`, `imRaw`, optionally `obj`, `initProbe`. |
| `location_file` + `raw_file` + `probe_file` (+ optional `gt_file`, `init_probe_file`) | Per-variable `.mat` files. |

Both MATLAB v5 (`scipy.io.loadmat`) and v7.3 / HDF5 formats are handled, including complex arrays stored as `{real, imag}` compound dtype.

Scan positions are physical (meters) and are projected onto the sample-plane pixel grid via

```
dx = wavelength * camera_length / (probe_shape * camera_pixel_pitch)
```

## Configuration

Everything is config-driven. Key knobs:

```yaml
model:
  downsample_factor: 2           # object generated at canvas/2, Fourier-upsampled
  log2_hashmap_size: 18
  hidden_features: 64
  hidden_layers: 2
  first_omega_0: 10.0            # exp(i·ω·Wx) activation scale
  hidden_omega_0: 1.0
  init_type: initNone            # initNone | initProbe | initQuadratic
  use_residual: true             # output = k*mlp + initial_pattern; k starts 1e-8

training:
  iters: 2000
  batch_size: 10
  lr: 1.0e-3
  loss_type: smooth_L1_loss      # smooth_L1_loss | FD_loss | Poisson_likelihood_loss
  use_amp: true                  # bfloat16 autocast
  sparsity_weight: 0.0           # Haar wavelet high-freq L1 penalty
```

`init_type=initProbe` + `use_residual: true` starts training from the measured probe as a prior and learns corrections on top — converges fastest. `initNone` trains from scratch and plateaus much higher (on plant, ~1e3 after 2000 epochs vs. <1e2 with a probe prior).

## Repository layout

```
Recovery.py          Entry point — loads config, builds model, runs loop
models/complex_inr.py  ComplexINRModel2D (HashGrid + complex Euler MLP)
utils/               io.py (mat v5/v7.3), optics.py (geometry helpers), loss.py
config/*.yaml        Per-dataset configs
legacy/              Reference code from prior experiments
CLAUDE.md            Architecture + gotchas for contributors
```

## License

Research code — no license specified.
