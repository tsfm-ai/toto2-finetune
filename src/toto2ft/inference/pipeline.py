from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.schema import DataSchema
from ..data.windowing import WindowDataset, collate_windows
from .quantiles import sort_quantiles, QUANTILE_LEVELS

logger = logging.getLogger(__name__)

_N_QUANTILES = len(QUANTILE_LEVELS)


class Toto2Pipeline:
    """
    Inference wrapper: given a (fine-tuned or zero-shot) Toto2Model,
    run walk-forward prediction over a DataFrame and return quantile forecasts.

    API mirrors Chronos 2's pipeline ergonomics so experiments can swap models
    by swapping pipelines.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    def predict(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        context_length: int,
        prediction_length: int,
        batch_size: int = 32,
        stride: int | None = None,
    ) -> np.ndarray:
        """
        Run inference over df.

        stride: defaults to prediction_length (non-overlapping windows).

        Returns q_preds: np.ndarray (Q, N, n_target, H)
          Q = 9 quantile levels
          N = number of forecast windows
        """
        if stride is None:
            stride = prediction_length

        dataset = WindowDataset(
            df=df,
            schema=schema,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_windows,
        )

        all_q_preds: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                target = batch["target"].to(self.device, dtype=self.dtype)
                target_mask = batch["target_mask"].to(self.device)
                cpm_mask = batch["cpm_mask"].to(self.device)
                series_ids = batch["series_ids"].to(self.device)

                outputs = self.model.forward(
                    target=target,
                    target_mask=target_mask,
                    cpm_mask=cpm_mask,
                    series_ids=series_ids,
                    num_return_steps=prediction_length,
                )

                n_target = len(schema.target_cols)
                q_pred = outputs.quantiles
                if q_pred.shape[0] == _N_QUANTILES:
                    q_target = q_pred[:, :, :n_target, :]
                else:
                    q_target = q_pred[:, :n_target, :, :].permute(2, 0, 1, 3)

                all_q_preds.append(sort_quantiles(q_target.cpu().numpy()))

        return np.concatenate(all_q_preds, axis=1)   # (Q, N, n_target, H)

    # ------------------------------------------------------------------
    def save(self, output_dir: str | Path) -> None:
        from ..training.checkpointing import save_checkpoint
        save_checkpoint(self.model, output_dir, step=-1)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Toto2Pipeline":
        """Load a zero-shot Toto 2 pipeline directly from HuggingFace."""
        from ..model.loader import load_toto2
        model = load_toto2(model_id, revision=revision, device=device or "cpu", dtype=dtype)
        return cls(model, device=device, dtype=dtype)

    @classmethod
    def from_checkpoint(
        cls,
        base_model_id: str,
        checkpoint_dir: str | Path,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Toto2Pipeline":
        """Reload a fine-tuned pipeline: loads base weights then applies saved adapter."""
        from ..model.loader import load_toto2
        from ..training.checkpointing import load_checkpoint
        model = load_toto2(base_model_id, device="cpu", dtype=dtype)
        model, _ = load_checkpoint(model, checkpoint_dir)
        return cls(model, device=device, dtype=dtype)
