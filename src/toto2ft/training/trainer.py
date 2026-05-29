from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
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
        # Unfreeze any output / head / quantile layers
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
        config.finetune_mode,
        f"{n_train:,}",
        f"{n_total:,}",
    )
    return model


def _cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    model: nn.Module,
    train_loader: DataLoader,
    config: TrainConfig,
    val_loader: DataLoader | None = None,
) -> list[dict]:
    """
    Fine-tune a Toto 2 model.

    Returns a training log: list of dicts with keys
      step, train_loss, lr, [val_loss].
    """
    torch.manual_seed(config.seed)

    device = _resolve_device(config)
    model = _setup_finetune_mode(model, config)
    model = model.to(device)

    quantiles_t = torch.tensor(QUANTILE_LEVELS, dtype=config.torch_dtype, device=device)
    n_quantiles = len(QUANTILE_LEVELS)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters — check finetune_mode and LoRA config.")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
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

            target = batch["target"].to(device, dtype=config.torch_dtype)
            target_mask = batch["target_mask"].to(device)
            cpm_mask = batch["cpm_mask"].to(device)
            series_ids = batch["series_ids"].to(device)
            future_values = batch["future_values"].to(device, dtype=config.torch_dtype)
            loss_mask = batch["loss_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model.forward(
                target=target,
                target_mask=target_mask,
                cpm_mask=cpm_mask,
                series_ids=series_ids,
                num_return_steps=config.prediction_length,
            )

            q_pred = outputs.quantiles   # (Q, B, V, H) or (B, V, Q, H)

            # Slice to target variates (first n_target) before computing loss
            n_target = future_values.shape[1]
            if q_pred.shape[0] == n_quantiles:
                q_pred_target = q_pred[:, :, :n_target, :]          # (Q, B, tV, H)
            else:
                q_pred_target = q_pred[:, :n_target, :, :].permute(2, 0, 1, 3)

            loss = pinball_loss(q_pred_target, future_values, quantiles_t, loss_mask)
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
                val_loss = _eval_loss(model, val_loader, config, device, quantiles_t, n_quantiles)
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
    n_quantiles: int,
) -> float:
    model.eval()
    losses = []
    for batch in val_loader:
        target = batch["target"].to(device, dtype=config.torch_dtype)
        target_mask = batch["target_mask"].to(device)
        cpm_mask = batch["cpm_mask"].to(device)
        series_ids = batch["series_ids"].to(device)
        future_values = batch["future_values"].to(device, dtype=config.torch_dtype)
        loss_mask = batch["loss_mask"].to(device)

        outputs = model.forward(
            target=target,
            target_mask=target_mask,
            cpm_mask=cpm_mask,
            series_ids=series_ids,
            num_return_steps=config.prediction_length,
        )

        n_target = future_values.shape[1]
        q_pred = outputs.quantiles
        if q_pred.shape[0] == n_quantiles:
            q_pred_target = q_pred[:, :, :n_target, :]
        else:
            q_pred_target = q_pred[:, :n_target, :, :].permute(2, 0, 1, 3)

        loss = pinball_loss(q_pred_target, future_values, quantiles_t, loss_mask)
        losses.append(loss.item())

    model.train()
    return float(np.mean(losses))
