from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model.lora import apply_lora
from ..model.losses import pinball_loss, QUANTILE_LEVELS
from .configs import TrainConfig

logger = logging.getLogger(__name__)


def _resolve_device(config: TrainConfig) -> torch.device:
    if config.device:
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _setup_finetune_mode(model: nn.Module, config: TrainConfig) -> nn.Module:
    if config.finetune_mode == "full":
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)

    elif config.finetune_mode == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        model = apply_lora(model, config.lora)
        model.train()

    elif config.finetune_mode == "head":
        for p in model.parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if any(kw in name for kw in ("output", "head", "quantile", "proj_out")):
                p.requires_grad_(True)
        model.train()

    else:
        raise ValueError(f"Unknown finetune_mode: {config.finetune_mode!r}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "mode=%s  trainable=%s/%s params",
        config.finetune_mode, f"{n_train:,}", f"{n_total:,}",
    )
    return model


def _cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _pad_to_patch(
    target: torch.Tensor,       # (B, V, C+H)
    target_mask: torch.Tensor,  # (B, V, C+H)
    cpm_mask: torch.Tensor,     # (B, V, C+H)
    pad_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad tensors to align future with patch boundary."""
    if pad_len == 0:
        return target, target_mask, cpm_mask
    target = F.pad(target, (0, pad_len))
    target_mask = F.pad(target_mask.float(), (0, pad_len)).bool()
    cpm_mask = F.pad(cpm_mask.float(), (0, pad_len)).bool()
    return target, target_mask, cpm_mask


def _unpack_outputs(
    outputs,
    n_target: int,
    prediction_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpack Toto2ModelOutputs for the prediction window.

    outputs.quantiles : (Q, B, V, n_patches, patch_size)  — normalized space
    outputs.loc       : (B, V, H_pad)
    outputs.scale     : (B, V, H_pad)

    Returns:
      q_norm : (Q, B, n_target, H)  predictions in normalized space
      loc    : (B, n_target, H)
      scale  : (B, n_target, H)
    """
    Q, B, V, n_patches, ps = outputs.quantiles.shape
    H_pad = n_patches * ps
    q_flat = outputs.quantiles.reshape(Q, B, V, H_pad)
    q_norm = q_flat[:, :, :n_target, :prediction_length]       # (Q, B, n_target, H)
    loc    = outputs.loc[:, :n_target, :prediction_length]     # (B, n_target, H)
    scale  = outputs.scale[:, :n_target, :prediction_length]   # (B, n_target, H)
    return q_norm, loc, scale


def train(
    model: nn.Module,
    train_loader: DataLoader,
    config: TrainConfig,
    val_loader: DataLoader | None = None,
) -> list[dict]:
    """
    Fine-tune a Toto 2 model.

    Handles patch-size alignment internally: context must be a multiple of
    patch_size (validated), horizon is padded to the next patch boundary.

    Returns a training log: list of dicts {step, train_loss, lr, [val_loss]}.
    """
    config.validate_geometry()
    torch.manual_seed(config.seed)

    H = config.prediction_length
    pad_len = config.prediction_length_padded - H  # bytes to add to right of future
    n_return_patches = config.n_return_patches

    device = _resolve_device(config)
    logger.info("Device: %s", device)

    model = _setup_finetune_mode(model, config)
    model = model.to(device)

    quantiles_t = torch.tensor(QUANTILE_LEVELS, dtype=config.torch_dtype, device=device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters — check finetune_mode and LoRA config.")

    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    log: list[dict] = []
    step = 0

    while step < config.num_steps:
        for batch in train_loader:
            if step >= config.num_steps:
                break

            lr = _cosine_lr(step, config.warmup_steps, config.num_steps, config.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            target      = batch["target"].to(device, dtype=config.torch_dtype)
            target_mask = batch["target_mask"].to(device)
            cpm_mask    = batch["cpm_mask"].to(device)
            series_ids  = batch["series_ids"].to(device)
            future_vals = batch["future_values"].to(device, dtype=config.torch_dtype)  # (B, tV, H)
            loss_mask   = batch["loss_mask"].to(device)                                # (B, tV, H)

            # Pad future to patch boundary (no-op if H is already aligned)
            target, target_mask, cpm_mask = _pad_to_patch(
                target, target_mask, cpm_mask, pad_len
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = model.forward(
                target=target,
                target_mask=target_mask,
                cpm_mask=cpm_mask,
                series_ids=series_ids,
                num_return_steps=n_return_patches,
            )

            n_target = future_vals.shape[1]
            q_norm, loc, scale = _unpack_outputs(outputs, n_target, H)

            # Compute loss in the model's normalized space to match pretraining regime.
            # Normalize targets with the same loc/scale the model used for context scaling.
            future_norm = (future_vals - loc) / scale.clamp_min(1e-6)

            loss = pinball_loss(q_norm, future_norm, quantiles_t, loss_mask)
            loss.backward()

            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable, config.grad_clip)

            optimizer.step()
            step += 1

            entry: dict = {"step": step, "train_loss": loss.item(), "lr": lr}

            if step % config.log_every == 0:
                logger.info(
                    "step=%d/%d  loss=%.4f  lr=%.2e",
                    step, config.num_steps, loss.item(), lr,
                )

            if val_loader and step % config.eval_every == 0:
                val_loss = _eval_loss(
                    model, val_loader, config, device, quantiles_t, pad_len, n_return_patches
                )
                entry["val_loss"] = val_loss
                logger.info("         val_loss=%.4f", val_loss)

            log.append(entry)

    return log


@torch.no_grad()
def _eval_loss(
    model: nn.Module,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    quantiles_t: torch.Tensor,
    pad_len: int,
    n_return_patches: int,
) -> float:
    model.eval()
    losses = []
    H = config.prediction_length
    for batch in val_loader:
        target      = batch["target"].to(device, dtype=config.torch_dtype)
        target_mask = batch["target_mask"].to(device)
        cpm_mask    = batch["cpm_mask"].to(device)
        series_ids  = batch["series_ids"].to(device)
        future_vals = batch["future_values"].to(device, dtype=config.torch_dtype)
        loss_mask   = batch["loss_mask"].to(device)

        target, target_mask, cpm_mask = _pad_to_patch(
            target, target_mask, cpm_mask, pad_len
        )

        outputs = model.forward(
            target=target,
            target_mask=target_mask,
            cpm_mask=cpm_mask,
            series_ids=series_ids,
            num_return_steps=n_return_patches,
        )

        n_target = future_vals.shape[1]
        q_norm, loc, scale = _unpack_outputs(outputs, n_target, H)
        future_norm = (future_vals - loc) / scale.clamp_min(1e-6)
        loss = pinball_loss(q_norm, future_norm, quantiles_t, loss_mask)
        losses.append(loss.item())

    model.train()
    return float(np.mean(losses))
