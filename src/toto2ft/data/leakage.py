from __future__ import annotations

import pandas as pd


def audit_availability(
    df: pd.DataFrame,
    feature_cols: list[str],
    availability_col: str,
    forecast_creation_col: str,
) -> pd.DataFrame:
    """
    Return rows where a feature's availability timestamp is AFTER forecast creation time.
    Output should be empty for a clean (non-leaky) dataset.
    """
    violations = df[df[availability_col] > df[forecast_creation_col]][
        [forecast_creation_col, availability_col] + feature_cols
    ]
    return violations.copy()


def flag_leaky_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    availability_col: str,
    horizon_start_col: str,
) -> list[str]:
    """
    Identify columns that have any non-null values where availability > horizon_start.
    A non-empty result means that column could encode realized future data.
    """
    future_mask = df[availability_col] > df[horizon_start_col]
    return [col for col in feature_cols if df.loc[future_mask, col].notna().any()]


def check_future_covariate_availability(
    df: pd.DataFrame,
    future_covariate_cols: list[str],
    timestamp_col: str,
    availability_col: str,
    prediction_length_hours: int,
    frequency: str = "1h",
) -> dict[str, bool]:
    """
    For each known-future covariate, verify that it is available at least
    prediction_length ahead of the timestamp it covers.

    Returns {col: True if clean, False if potential leakage detected}.
    """
    offset = pd.tseries.frequencies.to_offset(frequency)
    horizon_delta = prediction_length_hours * offset

    results = {}
    for col in future_covariate_cols:
        # For known-future cols: availability should be at or before timestamp - horizon
        required_available_by = df[timestamp_col] - horizon_delta
        leaky_rows = df[df[availability_col] > required_available_by][col].notna().sum()
        results[col] = leaky_rows == 0

    return results
