from __future__ import annotations

import torch


def build_masks(
    context_values: torch.Tensor,
    future_values: torch.Tensor,
    future_known_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build target_mask and cpm_mask for Toto 2's forward().

    context_values:   (B, V, C)  may contain NaN
    future_values:    (B, V, H)  may contain NaN
    future_known_mask (B, K, H)  bool — True for known-future covariate positions
                                  (K variates, positioned at the END of V)

    Returns:
      target_mask: (B, V, C+H)  True where a real value exists
      cpm_mask:    (B, V, C+H)  True where the model can observe the value
    """
    B, V, C = context_values.shape
    H = future_values.shape[-1]
    device = context_values.device

    ctx_obs = ~torch.isnan(context_values)   # (B, V, C)
    fut_obs = ~torch.isnan(future_values)    # (B, V, H)

    target_mask = torch.cat([ctx_obs, fut_obs], dim=-1)  # (B, V, C+H)

    # Context: visible if observed
    # Future targets: always hidden
    # Future known covariates: visible where observed
    fut_vis = torch.zeros(B, V, H, dtype=torch.bool, device=device)
    if future_known_mask is not None:
        K = future_known_mask.shape[1]
        fut_vis[:, -K:, :] = future_known_mask

    cpm_mask = torch.cat([ctx_obs, fut_vis], dim=-1)  # (B, V, C+H)

    return target_mask, cpm_mask
