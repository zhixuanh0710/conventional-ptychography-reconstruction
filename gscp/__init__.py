"""GSCP - 2D Gaussian Splatting rasterizer for coded ptychography.

Trimmed to only what the coded-ptychography-reconstruction project needs:
the ``GaussianFieldModel`` class and its CUDA rasterizer.
"""

from __future__ import annotations

from gscp.models import GaussianFieldModel

__all__ = ["GaussianFieldModel"]
