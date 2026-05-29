"""
Zero-shot vs fine-tuned comparison harness.

Usage::

    from toto2ft.evaluation.compare import compare_zero_shot_vs_ft

    results = compare_zero_shot_vs_ft(
        ft_model=model,          # fine-tuned nn.Module
        base_model_id="Datadog/Toto-2.0-313M",
        test_df=test_df,
        schema=schema,
        context_length=672,
        prediction_length=24,
        device=torch.device("mps"),
    )

    print(results["summary"])
    # WQL delta: -12.4%  MAE delta: -8.7%  (negative = fine-tuned wins)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..data.schema import DataSchema
from .backtest import walk_forward_backtest

logger = logging.getLogger(__name__)


def compare_zero_shot_vs_ft(
    ft_model: nn.Module,
    base_model_id: str,
    test_df: pd.DataFrame,
    schema: DataSchema,
    context_length: int,
    prediction_length: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 32,
    stride: int | None = None,
) -> dict:
    """
    Run walk-forward backtest for both zero-shot and fine-tuned models,
    return a result dict with metrics and % deltas.

    Returns:
      {
        "zero_shot":  {metric: value, ...},
        "fine_tuned": {metric: value, ...},
        "delta_pct":  {metric: pct_change, ...},   # negative = FT wins
        "summary":    str,                           # human-readable 1-liner
      }
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    # ── Zero-shot baseline ─────────────────────────────────────────────────────
    logger.info("Running zero-shot baseline with %s …", base_model_id)
    from ..model.loader import load_toto2
    zs_model = load_toto2(base_model_id, device=str(device), dtype=dtype)

    zs_metrics, _, _ = walk_forward_backtest(
        model=zs_model,
        df=test_df,
        schema=schema,
        context_length=context_length,
        prediction_length=prediction_length,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        stride=stride or prediction_length,
    )

    # Free zero-shot weights before fine-tuned run
    del zs_model
    if str(device) == "cuda":
        torch.cuda.empty_cache()

    # ── Fine-tuned ─────────────────────────────────────────────────────────────
    logger.info("Running fine-tuned model …")
    ft_metrics, _, _ = walk_forward_backtest(
        model=ft_model,
        df=test_df,
        schema=schema,
        context_length=context_length,
        prediction_length=prediction_length,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        stride=stride or prediction_length,
    )

    # ── Delta ──────────────────────────────────────────────────────────────────
    primary = ["WQL", "MAE", "RMSE", "sMAPE", "P10_P90_coverage", "interval_sharpness"]
    delta_pct: dict[str, float] = {}
    for k in primary:
        if k in zs_metrics and k in ft_metrics and zs_metrics[k] != 0:
            delta_pct[k] = (ft_metrics[k] - zs_metrics[k]) / abs(zs_metrics[k]) * 100.0

    summary_parts = []
    for k in ["WQL", "MAE"]:
        if k in delta_pct:
            sign = "+" if delta_pct[k] > 0 else ""
            summary_parts.append(f"{k} {sign}{delta_pct[k]:.1f}%")
    summary = "  |  ".join(summary_parts) + "  (negative = FT wins)"

    logger.info("Zero-shot vs FT: %s", summary)

    return {
        "zero_shot":  zs_metrics,
        "fine_tuned": ft_metrics,
        "delta_pct":  delta_pct,
        "summary":    summary,
    }


def print_comparison_table(results: dict) -> None:
    """Pretty-print zero-shot / fine-tuned side-by-side."""
    zs = results["zero_shot"]
    ft = results["fine_tuned"]
    dp = results["delta_pct"]

    width = 28
    print(f"\n{'Metric':<{width}} {'Zero-shot':>12} {'Fine-tuned':>12} {'Δ%':>8}")
    print("-" * (width + 36))
    for k in sorted(zs.keys()):
        zv = zs.get(k, float("nan"))
        fv = ft.get(k, float("nan"))
        dv = dp.get(k, float("nan"))
        d_str = f"{dv:+.1f}%" if not np.isnan(dv) else ""
        # Coverage metrics: higher is better, others: lower is better
        marker = ""
        if k in dp:
            wins = dv < 0 if k not in ("P10_P90_coverage",) else dv > 0
            marker = " ✓" if wins else ""
        print(f"{k:<{width}} {zv:>12.4f} {fv:>12.4f} {d_str:>8}{marker}")
    print()
    print(f"  → {results['summary']}")
