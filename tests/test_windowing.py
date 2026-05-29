import numpy as np
import pandas as pd
import torch
import pytest
from toto2ft.data.schema import DataSchema
from toto2ft.data.windowing import WindowDataset, collate_windows


@pytest.fixture
def future_cov_df():
    """DataFrame with both past and future covariates."""
    rng = np.random.default_rng(7)
    n = 200
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    records = []
    for zone in ["A"]:
        load = 1000 + rng.normal(0, 50, n)
        temp_act = 15 + rng.normal(0, 2, n)
        temp_fcast = temp_act + rng.normal(0, 0.5, n)  # slightly different
        hour_sin = np.sin(np.arange(n) * 2 * np.pi / 24)
        for i, t in enumerate(ts):
            records.append({
                "timestamp": t,
                "iso": "X",
                "zone": zone,
                "load": float(load[i]),
                "temperature_actual": float(temp_act[i]),
                "temperature_forecast": float(temp_fcast[i]),
                "hour_sin": float(hour_sin[i]),
            })
    return pd.DataFrame(records)


@pytest.fixture
def future_cov_schema():
    return DataSchema(
        timestamp_col="timestamp",
        group_cols=["iso", "zone"],
        target_cols=["load"],
        past_covariate_cols=["temperature_actual"],
        future_covariate_cols=["temperature_forecast", "hour_sin"],
        frequency="1h",
    )


def test_dataset_nonempty(sample_df, sample_schema):
    ds = WindowDataset(sample_df, sample_schema, context_length=48, prediction_length=24)
    assert len(ds) > 0


def test_sample_shapes(sample_df, sample_schema):
    C, H = 48, 24
    ds = WindowDataset(sample_df, sample_schema, context_length=C, prediction_length=H)
    s = ds[0]
    V = sample_schema.n_variates
    n_target = sample_schema.n_targets

    assert s.target.shape == (V, C + H)
    assert s.target_mask.shape == (V, C + H)
    assert s.cpm_mask.shape == (V, C + H)
    assert s.series_ids.shape == (V,)
    assert s.future_values.shape == (n_target, H)
    assert s.loss_mask.shape == (n_target, H)


def test_cpm_future_targets_hidden(sample_df, sample_schema):
    C, H = 48, 24
    ds = WindowDataset(sample_df, sample_schema, context_length=C, prediction_length=H)
    s = ds[0]
    n_target = sample_schema.n_targets
    # Future target variates must be hidden in cpm_mask
    fut_cpm_target = s.cpm_mask[:n_target, C:]
    assert not fut_cpm_target.any(), "Future target variates must be hidden (cpm_mask=False)"


def test_target_no_nan(sample_df, sample_schema):
    ds = WindowDataset(sample_df, sample_schema, context_length=48, prediction_length=24)
    for s in ds:
        assert not torch.isnan(s.target).any(), "target should have NaN replaced with 0"


def test_collate_stacks_batch_dim(sample_df, sample_schema):
    ds = WindowDataset(sample_df, sample_schema, context_length=48, prediction_length=24)
    B = min(4, len(ds))
    samples = [ds[i] for i in range(B)]
    batch = collate_windows(samples)
    assert batch["target"].shape[0] == B
    assert batch["future_values"].shape[0] == B


def test_target_only_schema(sample_df, target_only_schema):
    ds = WindowDataset(sample_df, target_only_schema, context_length=48, prediction_length=24)
    assert len(ds) > 0
    s = ds[0]
    assert s.target.shape == (1, 48 + 24)  # 1 variate (load only)


# ── Future-covariate CPM mask tests ──────────────────────────────────────────

def test_future_covariates_visible_in_horizon(future_cov_df, future_cov_schema):
    """Known-future covariates must be visible (cpm_mask=True) in the horizon."""
    C, H = 48, 24
    ds = WindowDataset(future_cov_df, future_cov_schema, context_length=C, prediction_length=H)
    assert len(ds) > 0
    s = ds[0]

    # Schema: [load(0), temperature_actual(1), temperature_forecast(2), hour_sin(3)]
    # future_covariate_indices should be [2, 3]
    fut_cov_idx = future_cov_schema.future_covariate_indices  # [2, 3]
    assert len(fut_cov_idx) == 2

    # Future known covariates must be visible where they are observed
    for idx in fut_cov_idx:
        fut_region = s.cpm_mask[idx, C:]          # (H,) — the horizon slice
        obs_region = s.target_mask[idx, C:]       # (H,) — observed where not NaN
        # Every observed known-future covariate must be visible in cpm_mask
        assert (fut_region[obs_region] == True).all(), (  # noqa: E712
            f"Future covariate at variate {idx} should be visible in horizon"
        )


def test_past_covariates_hidden_in_horizon(future_cov_df, future_cov_schema):
    """Past-observed covariates must be hidden (cpm_mask=False) in the horizon."""
    C, H = 48, 24
    ds = WindowDataset(future_cov_df, future_cov_schema, context_length=C, prediction_length=H)
    s = ds[0]

    # Past covariate: temperature_actual is at variate index 1
    n_target = future_cov_schema.n_targets
    n_past = len(future_cov_schema.past_covariate_cols)
    past_idx = list(range(n_target, n_target + n_past))  # [1]

    for idx in past_idx:
        fut_region = s.cpm_mask[idx, C:]   # (H,)
        assert not fut_region.any(), (
            f"Past covariate at variate {idx} must be hidden in horizon"
        )


def test_future_covariate_shape(future_cov_df, future_cov_schema):
    """Check variate count matches schema: 1 target + 1 past + 2 future = 4."""
    ds = WindowDataset(future_cov_df, future_cov_schema, context_length=48, prediction_length=24)
    s = ds[0]
    expected_V = future_cov_schema.n_variates  # 4
    assert s.target.shape == (expected_V, 48 + 24), f"Expected ({expected_V}, 72), got {s.target.shape}"
    assert s.future_values.shape == (1, 24), "future_values should only contain target variate"
    assert s.loss_mask.shape == (1, 24)
