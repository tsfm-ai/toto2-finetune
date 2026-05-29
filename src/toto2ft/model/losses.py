from __future__ import annotations

import torch

# Nine quantile levels matching Toto 2's output head
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_QUANTILES = len(QUANTILE_LEVELS)


def _to_qbvh(
    pred: torch.Tensor,
    n_quantiles: int,
) -> torch.Tensor:
    """
    Normalize quantile prediction tensor to (Q, B, V, H) layout.

    Toto 2 may return (Q, B, V, H) or (B, V, Q, H); handle both.
    """
    if pred.shape[0] == n_quantiles:
        return pred          # already (Q, B, V, H)
    # Assume (B, V, Q, H)
    return pred.permute(2, 0, 1, 3).contiguous()


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float] | torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean pinball loss over all unmasked positions and quantile levels.

    pred:      (Q, B, V, H) or (B, V, Q, H)
    target:    (B, V, H)
    quantiles: length-Q sequence
    loss_mask: (B, V, H)  True where loss is computed
    """
    if not isinstance(quantiles, torch.Tensor):
        quantiles = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)

    pred = _to_qbvh(pred, len(quantiles))                    # (Q, B, V, H)
    errors = target.unsqueeze(0) - pred                      # (Q, B, V, H)
    q = quantiles.view(-1, 1, 1, 1)
    loss = torch.maximum(q * errors, (q - 1.0) * errors)    # (Q, B, V, H)
    loss = loss * loss_mask.unsqueeze(0).float()
    denom = loss_mask.float().sum().clamp_min(1) * len(quantiles)
    return loss.sum() / denom


def weighted_pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float] | torch.Tensor,
    loss_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pinball loss with optional per-sample weights (e.g. upweight peak/event windows).

    sample_weights: (B,)
    """
    if not isinstance(quantiles, torch.Tensor):
        quantiles = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)

    pred = _to_qbvh(pred, len(quantiles))
    errors = target.unsqueeze(0) - pred
    q = quantiles.view(-1, 1, 1, 1)
    loss = torch.maximum(q * errors, (q - 1.0) * errors)
    loss = loss * loss_mask.unsqueeze(0).float()

    if sample_weights is not None:
        loss = loss * sample_weights.view(1, -1, 1, 1)

    denom = loss_mask.float().sum().clamp_min(1) * len(quantiles)
    return loss.sum() / denom
