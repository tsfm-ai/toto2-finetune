import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic energy-like DataFrame for unit tests (no model required)."""
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2024-01-01", periods=200, freq="1h")
    records = []
    for zone in ["A", "B"]:
        load = 1000 + 200 * np.sin(np.arange(200) * 2 * np.pi / 24) + rng.normal(0, 10, 200)
        temp = 15 + 5 * np.sin(np.arange(200) * 2 * np.pi / 24) + rng.normal(0, 1, 200)
        for i, ts in enumerate(timestamps):
            records.append({
                "timestamp": ts,
                "iso": "NYISO",
                "zone": zone,
                "load": float(load[i]),
                "temperature": float(temp[i]),
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_schema():
    from toto2ft.data.schema import DataSchema
    return DataSchema(
        timestamp_col="timestamp",
        group_cols=["iso", "zone"],
        target_cols=["load"],
        past_covariate_cols=["temperature"],
        frequency="1h",
    )


@pytest.fixture
def target_only_schema():
    from toto2ft.data.schema import DataSchema
    return DataSchema(
        timestamp_col="timestamp",
        group_cols=["iso", "zone"],
        target_cols=["load"],
        frequency="1h",
    )
