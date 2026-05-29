from __future__ import annotations

import numpy as np

QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
_P50_IDX = 4   # index of 0.5 in QUANTILE_LEVELS
_P10_IDX = 0
_P90_IDX = 8


def mae(actual: np.ndarray, median: np.ndarray) -> float:
    return float(np.nanmean(np.abs(actual - median)))


def rmse(actual: np.ndarray, median: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((actual - median) ** 2)))


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0 + 1e-8
    return float(np.nanmean(np.abs(actual - forecast) / denom) * 100.0)


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    seasonal_period: int = 24,
) -> float:
    naive_err = np.abs(np.diff(actual, n=seasonal_period)).mean() + 1e-8
    return float(np.abs(actual[seasonal_period:] - forecast[seasonal_period:]).mean() / naive_err)


def wql(
    actual: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantile_levels: np.ndarray = QUANTILE_LEVELS,
) -> float:
    """
    Mean pinball loss across all quantile levels.

    actual:             (...,)
    quantile_forecasts: (Q, ...) matching actual shape
    """
    total = 0.0
    for i, q in enumerate(quantile_levels):
        err = actual - quantile_forecasts[i]
        total += np.nanmean(np.maximum(q * err, (q - 1.0) * err))
    return total / len(quantile_levels)


def calibration_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    return float(np.mean((actual >= lower) & (actual <= upper)))


def interval_sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def compute_all_metrics(
    actual: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantile_levels: np.ndarray = QUANTILE_LEVELS,
    seasonal_period: int = 24,
) -> dict[str, float]:
    """
    Full metric suite from flat arrays.

    actual:             (N,)
    quantile_forecasts: (Q, N)
    """
    p10_idx = int(np.argmin(np.abs(quantile_levels - 0.1)))
    p50_idx = int(np.argmin(np.abs(quantile_levels - 0.5)))
    p90_idx = int(np.argmin(np.abs(quantile_levels - 0.9)))

    median = quantile_forecasts[p50_idx]
    lower = quantile_forecasts[p10_idx]
    upper = quantile_forecasts[p90_idx]

    return {
        "MAE":                  mae(actual, median),
        "RMSE":                 rmse(actual, median),
        "sMAPE":                smape(actual, median),
        "MASE":                 mase(actual, median, seasonal_period),
        "WQL":                  wql(actual, quantile_forecasts, quantile_levels),
        "P50_bias":             float(np.nanmean(median - actual)),
        # Empirical CDF values — well-calibrated model: P10≈0.10, P90≈0.90
        "P10_empirical_cdf":    float(np.nanmean(actual <= lower)),
        "P90_empirical_cdf":    float(np.nanmean(actual <= upper)),
        "P10_P90_coverage":     calibration_coverage(actual, lower, upper),
        "interval_sharpness":   interval_sharpness(lower, upper),
    }
