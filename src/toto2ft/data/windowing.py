from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .schema import DataSchema


@dataclass
class WindowSample:
    """One training sample in Toto 2 format (all tensors, no batch dim)."""
    target: torch.Tensor        # (V, C+H)  full sequence, NaN → 0 for model
    target_mask: torch.Tensor   # (V, C+H)  bool, True where value is real
    cpm_mask: torch.Tensor      # (V, C+H)  bool, True where model can see
    series_ids: torch.Tensor    # (V,)      int, same ID within one sample
    future_values: torch.Tensor # (n_target, H)  raw future targets for loss
    loss_mask: torch.Tensor     # (n_target, H)  bool, True where loss is computed


class WindowDataset(Dataset):
    """
    Rolling-window dataset from a long-format DataFrame.

    Builds (B, V, C+H) samples for Toto 2's forward():
      - context variates are visible to the model
      - future target variates are hidden (cpm_mask=False)
      - future known covariates remain visible (cpm_mask=True)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        context_length: int,
        prediction_length: int,
        stride: int = 1,
        min_context_frac: float = 0.5,
    ):
        schema.validate(df)
        self.schema = schema
        self.context_length = context_length
        self.prediction_length = prediction_length
        self._samples = list(
            self._build_windows(df, stride, min_context_frac)
        )

    # ------------------------------------------------------------------
    def _build_windows(
        self,
        df: pd.DataFrame,
        stride: int,
        min_context_frac: float,
    ) -> Iterator[WindowSample]:
        min_ctx = max(1, int(self.context_length * min_context_frac))
        C = self.context_length
        H = self.prediction_length

        all_cols = self.schema.all_value_cols
        n_all = len(all_cols)
        n_target = self.schema.n_targets
        future_covariate_idx = self.schema.future_covariate_indices  # indices into all_cols

        groups = df.groupby(self.schema.group_cols, sort=False)

        for _, group in groups:
            group = (
                group.sort_values(self.schema.timestamp_col)
                .reset_index(drop=True)
            )
            values = group[all_cols].to_numpy(dtype=np.float32)  # (T, V)
            T = len(values)

            for end in range(min_ctx + H, T + 1, stride):
                ctx_start = max(0, end - H - C)
                ctx_end = end - H

                context = values[ctx_start:ctx_end]   # (<= C, V)
                future = values[ctx_end:end]           # (H, V)

                ctx_len = len(context)
                if ctx_len < min_ctx:
                    continue

                # Left-pad context to C
                pad = C - ctx_len
                ctx_padded = np.full((C, n_all), np.nan, dtype=np.float32)
                ctx_padded[pad:] = context             # (C, V)

                # Stack into (V, C+H)
                full = np.concatenate([ctx_padded, future], axis=0).T  # (V, C+H)

                ctx_obs = ~np.isnan(ctx_padded).T      # (V, C) bool
                fut_obs = ~np.isnan(future).T           # (V, H) bool
                target_mask = np.concatenate([ctx_obs, fut_obs], axis=1)  # (V, C+H)

                # CPM: context visible if observed; future hidden for targets,
                # visible for known-future covariates
                fut_vis = np.zeros((n_all, H), dtype=bool)
                if future_covariate_idx:
                    fut_vis[future_covariate_idx, :] = fut_obs[future_covariate_idx, :]

                cpm_mask = np.concatenate([ctx_obs, fut_vis], axis=1)  # (V, C+H)

                series_ids = np.zeros(n_all, dtype=np.int64)

                # Loss: target variates only, future only, non-NaN only
                future_target = future[:, :n_target].T   # (n_target, H)
                loss_mask = ~np.isnan(future_target)

                yield WindowSample(
                    target=torch.from_numpy(np.nan_to_num(full, nan=0.0)),
                    target_mask=torch.from_numpy(target_mask),
                    cpm_mask=torch.from_numpy(cpm_mask),
                    series_ids=torch.from_numpy(series_ids),
                    future_values=torch.from_numpy(
                        np.nan_to_num(future_target, nan=0.0)
                    ),
                    loss_mask=torch.from_numpy(loss_mask),
                )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> WindowSample:
        return self._samples[idx]


def collate_windows(samples: list[WindowSample]) -> dict[str, torch.Tensor]:
    return {
        "target":        torch.stack([s.target for s in samples]),
        "target_mask":   torch.stack([s.target_mask for s in samples]),
        "cpm_mask":      torch.stack([s.cpm_mask for s in samples]),
        "series_ids":    torch.stack([s.series_ids for s in samples]),
        "future_values": torch.stack([s.future_values for s in samples]),
        "loss_mask":     torch.stack([s.loss_mask for s in samples]),
    }
