from __future__ import annotations

import numpy as np

QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def sort_quantiles(q_preds: np.ndarray) -> np.ndarray:
    """Enforce quantile monotonicity by sorting along the Q axis (axis 0)."""
    return np.sort(q_preds, axis=0)


def extract_median(
    q_preds: np.ndarray,
    quantile_levels: np.ndarray = QUANTILE_LEVELS,
) -> np.ndarray:
    idx = int(np.argmin(np.abs(quantile_levels - 0.5)))
    return q_preds[idx]


def quantiles_to_dict(
    q_preds: np.ndarray,
    quantile_levels: np.ndarray = QUANTILE_LEVELS,
) -> dict[str, np.ndarray]:
    """Map quantile predictions to named keys: q10, q20, ..., q90."""
    return {f"q{int(q * 100):02d}": q_preds[i] for i, q in enumerate(quantile_levels)}


def invert_scale(
    q_preds: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Undo robust scaling applied to model outputs.

    q_preds: (Q, ...) normalized predictions
    loc:     (...) or broadcastable
    scale:   (...) or broadcastable
    """
    return q_preds * scale[np.newaxis] + loc[np.newaxis]
