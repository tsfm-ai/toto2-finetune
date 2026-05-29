import torch
import pytest
from toto2ft.model.losses import pinball_loss, weighted_pinball_loss, QUANTILE_LEVELS

Q = len(QUANTILE_LEVELS)


def _make_pred(B, V, H, fill=0.0):
    return torch.full((Q, B, V, H), fill)


def test_pinball_loss_nonnegative():
    B, V, H = 2, 1, 24
    pred = _make_pred(B, V, H, 0.0)
    target = torch.ones(B, V, H)
    mask = torch.ones(B, V, H, dtype=torch.bool)
    loss = pinball_loss(pred, target, QUANTILE_LEVELS, mask)
    assert loss >= 0
    assert loss.isfinite()


def test_pinball_loss_zero_when_all_masked():
    B, V, H = 2, 1, 24
    pred = _make_pred(B, V, H, 999.0)
    target = torch.ones(B, V, H)
    mask = torch.zeros(B, V, H, dtype=torch.bool)
    loss = pinball_loss(pred, target, QUANTILE_LEVELS, mask)
    assert loss.item() == pytest.approx(0.0)


def test_pinball_loss_accepts_bvqh_shape():
    """(B, V, Q, H) shaped pred should be handled transparently."""
    B, V, H = 3, 2, 12
    pred_bvqh = torch.randn(B, V, Q, H)
    target = torch.randn(B, V, H)
    mask = torch.ones(B, V, H, dtype=torch.bool)
    loss = pinball_loss(pred_bvqh, target, QUANTILE_LEVELS, mask)
    assert loss.isfinite()
    assert loss >= 0


def test_pinball_loss_decreases_toward_correct_median():
    """Perfect P50 prediction should give lower loss than zero prediction."""
    B, V, H = 4, 1, 24
    target = torch.ones(B, V, H) * 5.0

    pred_zero = _make_pred(B, V, H, 0.0)
    pred_correct = _make_pred(B, V, H, 5.0)
    mask = torch.ones(B, V, H, dtype=torch.bool)

    loss_zero = pinball_loss(pred_zero, target, QUANTILE_LEVELS, mask)
    loss_correct = pinball_loss(pred_correct, target, QUANTILE_LEVELS, mask)
    assert loss_correct < loss_zero


def test_weighted_pinball_loss_higher_weight_increases_loss():
    B, V, H = 2, 1, 12
    pred = _make_pred(B, V, H, 0.0)
    target = torch.ones(B, V, H)
    mask = torch.ones(B, V, H, dtype=torch.bool)

    uniform_w = torch.ones(B)
    heavy_w = torch.tensor([1.0, 10.0])

    loss_uniform = weighted_pinball_loss(pred, target, QUANTILE_LEVELS, mask, uniform_w)
    loss_heavy = weighted_pinball_loss(pred, target, QUANTILE_LEVELS, mask, heavy_w)
    assert loss_heavy > loss_uniform
