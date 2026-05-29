import numpy as np
import pytest
from toto2ft.evaluation.metrics import (
    mae,
    rmse,
    wql,
    calibration_coverage,
    compute_all_metrics,
    QUANTILE_LEVELS,
)

Q = len(QUANTILE_LEVELS)


def test_mae_perfect():
    x = np.array([1.0, 2.0, 3.0])
    assert mae(x, x) == pytest.approx(0.0)


def test_rmse_perfect():
    x = np.array([1.0, 2.0, 3.0])
    assert rmse(x, x) == pytest.approx(0.0)


def test_wql_zero_on_perfect_forecasts():
    actual = np.ones(100)
    q_preds = np.ones((Q, 100))   # all quantiles == actual
    assert wql(actual, q_preds, QUANTILE_LEVELS) == pytest.approx(0.0, abs=1e-8)


def test_wql_nonnegative():
    rng = np.random.default_rng(1)
    actual = rng.normal(0, 1, 200)
    q_preds = rng.normal(0, 1, (Q, 200))
    assert wql(actual, q_preds, QUANTILE_LEVELS) >= 0


def test_calibration_perfect_coverage():
    actual = np.linspace(0, 1, 100)
    lower = np.zeros(100)
    upper = np.ones(100)
    assert calibration_coverage(actual, lower, upper) == pytest.approx(1.0)


def test_calibration_no_coverage():
    actual = np.zeros(100)
    lower = np.ones(100)
    upper = np.ones(100) * 2
    assert calibration_coverage(actual, lower, upper) == pytest.approx(0.0)


def test_compute_all_metrics_returns_expected_keys():
    actual = np.sin(np.linspace(0, 4 * np.pi, 200))
    q_preds = np.stack([actual + i * 0.01 for i in range(Q)])
    metrics = compute_all_metrics(actual, q_preds)
    expected_keys = {"MAE", "RMSE", "sMAPE", "MASE", "WQL", "P50_bias",
                     "P10_empirical_cdf", "P90_empirical_cdf", "P10_P90_coverage",
                     "interval_sharpness"}
    assert expected_keys.issubset(metrics.keys())
    for v in metrics.values():
        assert np.isfinite(v), f"Non-finite metric value: {v}"
