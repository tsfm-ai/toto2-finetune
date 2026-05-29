import torch
import pytest
from toto2ft.data.windowing import WindowDataset, collate_windows


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
