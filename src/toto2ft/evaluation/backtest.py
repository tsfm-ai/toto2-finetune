from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.schema import DataSchema
from ..data.windowing import WindowDataset, collate_windows
from .metrics import compute_all_metrics, QUANTILE_LEVELS

logger = logging.getLogger(__name__)

_N_QUANTILES = len(QUANTILE_LEVELS)


def walk_forward_backtest(
    model: nn.Module,
    df: pd.DataFrame,
    schema: DataSchema,
    context_length: int,
    prediction_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 32,
    stride: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """
    Walk-forward (non-overlapping by default) backtest.

    Returns:
      metrics:  dict of scalar metrics over the entire split
      actuals:  np.ndarray (N, n_target, H)
      q_preds:  np.ndarray (Q, N, n_target, H)
    """
    if stride is None:
        stride = prediction_length

    dataset = WindowDataset(
        df=df,
        schema=schema,
        context_length=context_length,
        prediction_length=prediction_length,
        stride=stride,
    )

    if len(dataset) == 0:
        raise ValueError("No valid windows in backtest split. Check date ranges and context_length.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_windows,
    )

    all_actuals: list[np.ndarray] = []
    all_q_preds: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device, dtype=dtype)
            target_mask = batch["target_mask"].to(device)
            cpm_mask = batch["cpm_mask"].to(device)
            series_ids = batch["series_ids"].to(device)

            outputs = model.forward(
                target=target,
                target_mask=target_mask,
                cpm_mask=cpm_mask,
                series_ids=series_ids,
                num_return_steps=prediction_length,
            )

            n_target = batch["future_values"].shape[1]
            q_pred = outputs.quantiles
            if q_pred.shape[0] == _N_QUANTILES:
                q_target = q_pred[:, :, :n_target, :]          # (Q, B, tV, H)
            else:
                q_target = q_pred[:, :n_target, :, :].permute(2, 0, 1, 3)

            all_actuals.append(batch["future_values"].cpu().numpy())  # (B, tV, H)
            all_q_preds.append(q_target.cpu().numpy())                # (Q, B, tV, H)

    actuals = np.concatenate(all_actuals, axis=0)   # (N, tV, H)
    q_preds = np.concatenate(all_q_preds, axis=1)   # (Q, N, tV, H)

    flat_actual = actuals.reshape(-1)
    flat_qpred = q_preds.reshape(_N_QUANTILES, -1)
    metrics = compute_all_metrics(flat_actual, flat_qpred)

    logger.info(
        "Backtest: N=%d windows  MAE=%.4f  WQL=%.4f  P10/P90_cov=%.3f",
        len(actuals), metrics["MAE"], metrics["WQL"], metrics["P10_P90_coverage"],
    )
    return metrics, actuals, q_preds
