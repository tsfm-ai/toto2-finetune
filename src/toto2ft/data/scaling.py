from __future__ import annotations

import numpy as np
import torch


def robust_scale(
    values: np.ndarray,
    eps: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Per-series robust scaling: (x - median) / (MAD + eps). Returns (scaled, loc, scale)."""
    loc = float(np.nanmedian(values))
    scale = float(np.nanmedian(np.abs(values - loc))) + eps
    return (values - loc) / scale, loc, scale


def robust_scale_tensor(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-variate robust instance normalization.

    x:    (B, V, T)
    mask: (B, V, T) bool, True where values are real (not padding/NaN)

    Returns (normalized, loc, scale) each with loc/scale shaped (B, V, 1).
    """
    working = x.clone().float()
    if mask is not None:
        working[~mask] = float("nan")

    loc = working.nanmedian(dim=-1, keepdim=True).values        # (B, V, 1)
    mad = (working - loc).abs().nanmedian(dim=-1, keepdim=True).values + eps  # (B, V, 1)

    loc = torch.nan_to_num(loc, nan=0.0)
    mad = torch.nan_to_num(mad, nan=eps).clamp_min(eps)

    normalized = (x - loc) / mad
    return normalized.to(x.dtype), loc.to(x.dtype), mad.to(x.dtype)
