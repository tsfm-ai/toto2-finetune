from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics, QUANTILE_LEVELS

_N_QUANTILES = len(QUANTILE_LEVELS)


def stratified_metrics(
    actuals: np.ndarray,
    q_preds: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute metrics stratified by columns in metadata.

    actuals:  (N, n_target, H)
    q_preds:  (Q, N, n_target, H)
    metadata: pd.DataFrame with N rows; columns are slice dimensions
              e.g. zone, season, is_holiday, hour_bucket

    Returns nested dict:  {column → {value → metrics_dict}}
    """
    results: dict = {}

    for col in metadata.columns:
        results[col] = {}
        for val in sorted(metadata[col].unique(), key=str):
            mask = (metadata[col].values == val)
            if mask.sum() == 0:
                continue
            flat_actual = actuals[mask].reshape(-1)
            flat_qpred = q_preds[:, mask, :, :].reshape(_N_QUANTILES, -1)
            results[col][str(val)] = compute_all_metrics(flat_actual, flat_qpred)

    return results


def horizon_step_metrics(
    actuals: np.ndarray,
    q_preds: np.ndarray,
) -> list[dict[str, float]]:
    """
    Per-horizon-step metrics to diagnose error accumulation.

    actuals:  (N, n_target, H)
    q_preds:  (Q, N, n_target, H)

    Returns list of length H, one metrics dict per horizon step.
    """
    H = actuals.shape[-1]
    out = []
    for h in range(H):
        flat_actual = actuals[:, :, h].reshape(-1)
        flat_qpred = q_preds[:, :, :, h].reshape(_N_QUANTILES, -1)
        out.append(compute_all_metrics(flat_actual, flat_qpred))
    return out
