"""
Loss functions for conventional ptychography reconstruction.
"""

import torch
from torch.nn import functional as F


def calculate_loss(captured, estimated, loss_type='smoothL1Loss', reduction='mean',
                   object_recovery=None, sparsity_weight=0.0,
                   im_size_x=None, im_size_y=None):
    """
    Loss between captured and simulated diffraction intensity.

    loss_type : 'smoothL1Loss' | 'l1Loss' | 'mseLoss' | 'GDLoss'
                | 'PoissonLoss'
    """
    eps = 1e-8
    if captured.dtype != estimated.dtype:
        captured = captured.to(torch.float32)
        estimated = estimated.to(torch.float32)

    if loss_type == 'l1Loss':
        supervised = F.l1_loss(captured, estimated, reduction=reduction)
    elif loss_type == 'smoothL1Loss':
        supervised = F.smooth_l1_loss(captured, estimated, reduction=reduction)
    elif loss_type == 'mseLoss':
        supervised = F.mse_loss(captured, estimated, reduction=reduction)
    elif loss_type == 'PoissonLoss':
        supervised = (estimated - captured * torch.log(estimated + eps))
        supervised = _reduce(supervised, reduction)
    elif loss_type == 'GDLoss':
        dw = torch.sqrt(captured + eps) - torch.sqrt(estimated + eps)
        rx_dw = torch.cat((dw[:, 1:] - dw[:, :-1], dw[:, :1] - dw[:, -1:]), dim=1)
        ry_dw = torch.cat((dw[1:, :] - dw[:-1, :], dw[:1, :] - dw[-1:, :]), dim=0)
        grad = torch.sqrt(rx_dw ** 2 + ry_dw ** 2 + eps)
        supervised = _reduce(grad, reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if object_recovery is not None and sparsity_weight > 0:
        sp = haar_wavelet_sparsity_loss(object_recovery)
        if reduction == 'mean' and im_size_x is not None and im_size_y is not None:
            sp = sp / (im_size_x * im_size_y)
        return supervised + sparsity_weight * sp

    return supervised


def _reduce(x, reduction):
    if reduction == 'mean':
        return x.mean()
    if reduction == 'sum':
        return x.sum()
    if reduction == 'none':
        return x
    raise ValueError(f"Unknown reduction: {reduction}")


def haar_wavelet_sparsity_loss(img, reduction='sum'):
    """L1 sparsity on Haar high-frequency subbands (LH, HL, HH).

    Default ``reduction='sum'`` matches the legacy notebook scale so YAML
    ``sparsity_weight`` values transfer directly.
    """
    if torch.is_complex(img):
        return (haar_wavelet_sparsity_loss(img.real, reduction)
                + haar_wavelet_sparsity_loss(img.imag, reduction))
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3:
        img = img.unsqueeze(1)

    a = img[:, :, 0::2, 0::2]
    b = img[:, :, 0::2, 1::2]
    c = img[:, :, 1::2, 0::2]
    d = img[:, :, 1::2, 1::2]

    lh = a - b + c - d
    hl = a + b - c - d
    hh = a - b - c + d
    reduce = torch.sum if reduction == 'sum' else torch.mean
    return reduce(torch.abs(lh)) + reduce(torch.abs(hl)) + reduce(torch.abs(hh))


def com_loss(probe):
    """Center-of-mass regularization on probe amplitude."""
    amp2 = probe.abs() ** 2
    total = amp2.sum() + 1e-8
    M, N = amp2.shape
    m = torch.arange(M, device=probe.device).view(M, 1)
    n = torch.arange(N, device=probe.device).view(1, N)
    x0 = (m * amp2).sum() / total
    y0 = (n * amp2).sum() / total
    cx = (M - 1) / 2
    cy = (N - 1) / 2
    return (x0 - cx) ** 2 + (y0 - cy) ** 2
