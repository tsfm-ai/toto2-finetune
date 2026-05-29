from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        patch_size: int = 32,
    ) -> np.ndarray:
        """
        Run inference over df.

        stride: defaults to prediction_length (non-overlapping windows).

        Returns q_preds: np.ndarray (Q, N, n_target, H)
          Q = 9 quantile levels  N = forecast windows
        """
        if stride is None:
            stride = prediction_length

        H_pad = math.ceil(prediction_length / patch_size) * patch_size
        pad_len = H_pad - prediction_length
        n_return_patches = H_pad // patch_size

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
                target      = batch["target"].to(self.device, dtype=self.dtype)
                target_mask = batch["target_mask"].to(self.device)
                cpm_mask    = batch["cpm_mask"].to(self.device)
                series_ids  = batch["series_ids"].to(self.device)

                if pad_len > 0:
                    target      = F.pad(target, (0, pad_len))
                    target_mask = F.pad(target_mask.float(), (0, pad_len)).bool()
                    cpm_mask    = F.pad(cpm_mask.float(), (0, pad_len)).bool()

                outputs = self.model.forward(
                    target=target,
                    target_mask=target_mask,
                    cpm_mask=cpm_mask,
                    series_ids=series_ids,
                    num_return_steps=n_return_patches,
                )

                n_target = len(schema.target_cols)
                # Unpack: (Q, B, V, n_patches, patch_size) → normalized (Q, B, n_target, H)
                q_raw = outputs.quantiles
                Q, B, V, n_p, ps = q_raw.shape
                H_pad_actual = n_p * ps
                q_flat = q_raw.reshape(Q, B, V, H_pad_actual)
                q_norm = q_flat[:, :, :n_target, :prediction_length]    # (Q, B, n_target, H)

                # Denormalize to original scale using model's scaler output
                loc   = outputs.loc[:, :n_target, :prediction_length]   # (B, n_target, H)
                scale = outputs.scale[:, :n_target, :prediction_length] # (B, n_target, H)
                q_real = q_norm * scale.unsqueeze(0) + loc.unsqueeze(0) # (Q, B, n_target, H)

                all_q_preds.append(sort_quantiles(q_real.cpu().numpy()))

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
