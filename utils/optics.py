"""
Optical propagation, forward imaging model, and geometry helpers
for conventional (far-field) ptychography.
"""

import math
from math import pi

import numpy as np
import torch


def propagate(field, pixel_size, wavelength, distance):
    """
    Angular spectrum propagation of a complex 2D field.
    """
    M, N = field.shape
    k0 = 2 * pi / wavelength
    device = field.device

    ky0 = torch.fft.fftshift(torch.fft.fftfreq(M, d=pixel_size, device=device)) * (2 * pi)
    kx0 = torch.fft.fftshift(torch.fft.fftfreq(N, d=pixel_size, device=device)) * (2 * pi)
    ky, kx = torch.meshgrid(ky0, kx0, indexing='ij')

    real_part = k0 ** 2 - kx ** 2 - ky ** 2
    kz = torch.sqrt(torch.complex(real_part, torch.zeros_like(real_part)))
    mask = (real_part >= 0).to(torch.complex64)
    hz = (torch.exp(1j * distance * torch.real(kz))
          * torch.exp(-abs(distance) * torch.abs(torch.imag(kz)))
          * mask)

    field_fft = torch.fft.fftshift(torch.fft.fft2(field))
    return torch.fft.ifft2(torch.fft.ifftshift(hz * field_fft))


def sub_pixel_shift(image, x_shift, y_shift, mag=1):
    """Sub-pixel shift via Fourier phase ramp."""
    m, n = image.shape
    fy = torch.linspace(-torch.floor(torch.tensor(m / 2)),
                        torch.ceil(torch.tensor(m / 2)) - 1,
                        m, device=image.device, dtype=image.dtype)
    fy = torch.fft.ifftshift(fy)
    fx = torch.linspace(-torch.floor(torch.tensor(n / 2)),
                        torch.ceil(torch.tensor(n / 2)) - 1,
                        n, device=image.device, dtype=image.dtype)
    fx = torch.fft.ifftshift(fx)
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')
    hs = torch.exp(-1j * 2 * pi * (FX * -x_shift / n * mag + FY * -y_shift / m * mag))
    return torch.fft.ifft2(torch.fft.fft2(image) * hs)


def freq_shift(field, shift):
    """
    Sub-pixel Fourier-domain translation.

    Parameters
    ----------
    field : complex tensor, shape (M, N)
    shift : tensor [Δx, Δy] in pixel units.
    """
    M, N = field.shape[-2], field.shape[-1]
    u = torch.fft.fftfreq(M, d=1.0, device=field.device).view(M, 1)
    v = torch.fft.fftfreq(N, d=1.0, device=field.device).view(1, N)
    phase = torch.exp(-2j * math.pi * (u * shift[0] + v * shift[1]))
    return torch.fft.ifft2(torch.fft.fft2(field) * phase)


def scan_positions_to_pixels(xlocation, ylocation, probe_shape,
                              wavelength, camera_length, camera_pixel_pitch):
    """
    Convert physical scan positions (meters) to sample-plane pixel indices.

    Returns
    -------
    tlX, tlY : int arrays
        Top-left pixel index for each scan position.
    brX, brY : int arrays
        Bottom-right pixel index (= top-left + probe size).
    dx : ndarray, shape (2,)
        Sample-plane pixel pitch along (y, x).
    canvas_size : int
        Minimum square canvas size, rounded up to a multiple of 6, that
        contains all probe footprints.
    """
    M, N = probe_shape
    x_pos = np.asarray(xlocation).squeeze() - np.min(xlocation)
    y_pos = np.asarray(ylocation).squeeze() - np.min(ylocation)

    dx = wavelength * camera_length / (np.array([M, N]) * camera_pixel_pitch)

    tlY = np.round(y_pos / dx[0]).astype(int)
    tlX = np.round(x_pos / dx[1]).astype(int)
    brY = tlY + M
    brX = tlX + N

    max_size = max(int(np.max(brY)), int(np.max(brX)))
    canvas_size = math.ceil(max_size / 6) * 6

    return tlX, tlY, brX, brY, dx, canvas_size


def forward_imaging_model(object_complex, probe_complex, tlY, tlX, brY, brX):
    """
    Conventional far-field ptychography forward model.

    Exit wave = probe * object[tlY:brY, tlX:brX]
    Measurement = |FFT(exit wave)|^2 (fftshift applied).
    """
    patch = object_complex[tlY:brY, tlX:brX]
    exit_wave = probe_complex * patch
    fft_ew = torch.fft.fftshift(torch.fft.fft2(exit_wave), dim=(-2, -1))
    return torch.abs(fft_ew) ** 2


def quadratic_phase_probe(probe, dx, wavelength, focal_length):
    """
    Apply a Fresnel quadratic phase factor to a complex probe.

    out = exp(i * pi / (lam * L) * (X^2 + Y^2)) * probe

    Parameters
    ----------
    probe : complex ndarray or torch tensor, shape (M, N)
    dx : float or 2-tuple
        Pixel pitch (m). Scalar uses same pitch for both axes.
    wavelength : float
    focal_length : float
        Effective propagation distance L (m).
    """
    is_torch = torch.is_tensor(probe)
    if is_torch:
        device = probe.device
        probe_arr = probe.detach().cpu().numpy()
    else:
        probe_arr = np.asarray(probe)

    M, N = probe_arr.shape
    dy = dx[0] if hasattr(dx, '__len__') else float(dx)
    dxx = dx[1] if hasattr(dx, '__len__') else float(dx)
    yp = (np.arange(M, dtype=np.float32) - M / 2) * dy
    xp = (np.arange(N, dtype=np.float32) - N / 2) * dxx
    X, Y = np.meshgrid(xp, yp)
    phase = np.pi / (wavelength * focal_length) * (X ** 2 + Y ** 2)
    out = np.exp(1j * phase).astype(np.complex64) * probe_arr.astype(np.complex64)

    if is_torch:
        return torch.from_numpy(out).to(device)
    return out


def center_probe(probe_complex):
    """
    Recenter the probe via center-of-mass-based circular roll.
    Returns the recentered probe (same dtype/shape).
    """
    # Force float32 so the divisions are safe even when the probe is
    # bfloat16 inside an autocast context.
    absP2 = (probe_complex.abs() ** 2).float()
    M, N = absP2.shape
    tot = absP2.sum() + 1e-12
    MN = torch.tensor([M, N], device=probe_complex.device, dtype=torch.float32)
    com = torch.stack((
        absP2.sum(1).cumsum(0).mean(),
        absP2.sum(0).cumsum(0).mean(),
    ))
    raw = MN / 2 - MN * com / tot + 1
    if not torch.isfinite(raw).all():
        return probe_complex
    cp = torch.trunc(raw.clamp(-M, M)).long()
    if cp.any():
        probe_complex = torch.roll(probe_complex, (-cp[0].item(), -cp[1].item()), (0, 1))
    return probe_complex
