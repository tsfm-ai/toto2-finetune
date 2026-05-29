import torch
from toto2ft.model.masks import build_masks


def test_shapes():
    B, V, C, H = 2, 3, 48, 24
    ctx = torch.randn(B, V, C)
    fut = torch.randn(B, V, H)
    tm, cm = build_masks(ctx, fut)
    assert tm.shape == (B, V, C + H)
    assert cm.shape == (B, V, C + H)


def test_future_always_hidden_without_known():
    B, V, C, H = 2, 3, 48, 24
    ctx = torch.randn(B, V, C)
    fut = torch.randn(B, V, H)
    _, cm = build_masks(ctx, fut)
    assert not cm[:, :, C:].any(), "All future positions hidden when no known-future mask"


def test_context_matches_observed():
    B, V, C, H = 1, 2, 24, 12
    ctx = torch.randn(B, V, C)
    ctx[0, 0, :4] = float("nan")    # 4 missing values in variate 0
    fut = torch.randn(B, V, H)
    tm, cm = build_masks(ctx, fut)
    # NaN positions → False in target_mask
    assert not tm[0, 0, :4].any()
    # CPM context visibility equals target_mask context part
    assert (cm[:, :, :C] == tm[:, :, :C]).all()


def test_known_future_visible():
    B, V, C, H = 1, 4, 24, 12
    ctx = torch.randn(B, V, C)
    fut = torch.randn(B, V, H)
    # 2 known-future covariates at the end of V
    known_mask = torch.ones(B, 2, H, dtype=torch.bool)
    _, cm = build_masks(ctx, fut, future_known_mask=known_mask)
    # Last 2 variates future should be visible
    assert cm[:, -2:, C:].all()
    # First 2 variates future should still be hidden
    assert not cm[:, :2, C:].any()
