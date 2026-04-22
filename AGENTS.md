# Repository Guidelines

## Project Structure & Module Organization
`Recovery.py` is the main training and reconstruction entrypoint. Core model code lives in `models/`, with `models/complex_inr.py` defining the complex INR used for both object and probe branches. Shared helpers are in `utils/` (`io.py`, `optics.py`, `loss.py`). Dataset-specific settings live in `config/*.yaml`. Use `scripts/simulate_and_reconstruct.py` for synthetic end-to-end validation. `legacy/` contains reference implementations; treat it as historical unless you are intentionally syncing behavior. `results/`, `Recovery.log`, `.mat`, and `.pth` outputs are generated artifacts and should not be committed.

## Build, Test, and Development Commands
Set up the Python environment with the packages listed in `README.md`:

```bash
conda create -n CP_torch python=3.10
conda activate CP_torch
pip install torch torchvision scipy pyyaml tqdm matplotlib h5py
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Run reconstructions with a config:

```bash
python Recovery.py --config config/plant.yaml
python Recovery.py --config config/clam.yaml
python Recovery.py --config config/simulation.yaml
python scripts/simulate_and_reconstruct.py
```

Only if you are working on `gscp/`, build its CUDA extension with `pip install gscp/csrc/`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for module constants, and concise docstrings where behavior is not obvious. Keep config keys aligned with existing YAML names such as `init_type`, `loss_type`, and `result_dir`. Prefer small utility functions in `utils/` over duplicating logic in `Recovery.py`.

## Testing Guidelines
There is no formal pytest suite yet. Validate changes by running at least one real config and, for algorithm changes, `python scripts/simulate_and_reconstruct.py`. Preserve output locations under `results/` and check for expected artifacts such as `result.mat`, `trained_models.pth`, and epoch PNGs. If you add tests later, place them under a new `tests/` directory and name files `test_*.py`.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Add README` and `Refactor Recovery.py setup to reuse utils helpers`. Keep commits focused and descriptive. PRs should summarize the reconstruction or data-loading behavior changed, list the config or script used for validation, and attach result images when the change affects outputs or training behavior.

## Configuration & Data Notes
Keep dataset paths and physics parameters in YAML, not hard-coded in Python. Do not commit raw datasets, generated checkpoints, logs, or notebook outputs. If you introduce new large artifacts, update `.gitignore` in the same change.
