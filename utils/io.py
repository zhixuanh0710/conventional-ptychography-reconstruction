"""
I/O utilities: .mat loading and model save/load.
"""

import os
import numpy as np
import torch
from scipy.io import loadmat


def save_model(model, save_path):
    """Save only requires_grad parameters."""
    tensors = [p for _, p in model.named_parameters() if p.requires_grad]
    torch.save(tensors, save_path)


def load_model(model, load_path):
    """Load requires_grad parameters from a saved list."""
    tensors = torch.load(load_path)
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = tensors.pop(0).data


def read_data(file_path, key_name=None):
    """
    Load MATLAB .mat file (legacy or v7.3 HDF5 format) with complex support.
    """
    try:
        return loadmat(file_path)
    except Exception:
        pass

    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is not installed. Run: pip install h5py")

    def _transpose(arr):
        if arr.ndim >= 2:
            return np.transpose(arr)
        return arr

    def _to_numpy(dset):
        raw = dset[()]
        if hasattr(raw, 'dtype') and raw.dtype.fields and \
                {'real', 'imag'} <= set(raw.dtype.fields.keys()):
            return _transpose(np.array(raw['real']) + 1j * np.array(raw['imag']))
        if dset.attrs.get('MATLAB_complex', False):
            arr = np.array(raw)
            if arr.shape[0] == 2:
                return _transpose(arr[0] + 1j * arr[1])
            if arr.shape[-1] == 2:
                return _transpose(arr[..., 0] + 1j * arr[..., 1])
        return _transpose(np.array(raw))

    def _convert(obj):
        if isinstance(obj, h5py.Dataset):
            return _to_numpy(obj)
        if isinstance(obj, h5py.Group):
            if 'real' in obj and 'imag' in obj:
                return _convert(obj['real']) + 1j * _convert(obj['imag'])
            return {k: _convert(obj[k]) for k in obj.keys() if not k.startswith('#')}
        return None

    with h5py.File(file_path, 'r') as f:
        if key_name and key_name in f:
            return {key_name: _convert(f[key_name])}
        return {k: _convert(f[k]) for k in f.keys() if not k.startswith('#')}


def load_ptychography_data(data_dir, bundle_file=None, location_file=None,
                            raw_file=None, probe_file=None, gt_file=None,
                            init_probe_file=None,
                            loc_x_key='xlocation', loc_y_key='ylocation',
                            raw_key='imRaw', probe_key='probe',
                            gt_key='obj', init_probe_key='initProbe',
                            gap=1):
    """
    Load a conventional ptychography dataset.

    Either pass ``bundle_file`` (a single .mat containing all variables)
    or the per-variable files. Individual files take precedence if both.

    Returns a dict with keys: xlocation, ylocation, probe, im_raw,
    optional obj (ground truth) and init_probe.
    """
    out = {}

    if bundle_file is not None:
        bundle = read_data(os.path.join(data_dir, bundle_file))
        for k in (loc_x_key, loc_y_key, probe_key, raw_key, gt_key, init_probe_key):
            if k in bundle:
                out[k] = bundle[k]

    if location_file is not None:
        loc = read_data(os.path.join(data_dir, location_file))
        out[loc_x_key] = loc[loc_x_key]
        out[loc_y_key] = loc[loc_y_key]
    if raw_file is not None:
        out[raw_key] = read_data(os.path.join(data_dir, raw_file))[raw_key]
    if probe_file is not None:
        out[probe_key] = read_data(os.path.join(data_dir, probe_file))[probe_key]
    if gt_file is not None:
        gt = read_data(os.path.join(data_dir, gt_file))
        out[gt_key] = gt.get(gt_key, next(iter(v for k, v in gt.items()
                                              if not k.startswith('__')), None))
    if init_probe_file is not None:
        ip = read_data(os.path.join(data_dir, init_probe_file))
        out[init_probe_key] = ip.get(init_probe_key,
                                     next(iter(v for k, v in ip.items()
                                               if not k.startswith('__')), None))

    out[loc_x_key] = np.asarray(out[loc_x_key]).squeeze()
    out[loc_y_key] = np.asarray(out[loc_y_key]).squeeze()

    if gap and gap > 1:
        out[loc_x_key] = out[loc_x_key][::gap]
        out[loc_y_key] = out[loc_y_key][::gap]
        out[raw_key] = out[raw_key][:, :, ::gap]

    return out
