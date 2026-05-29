from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # None → auto-discover attention projections from model
    target_modules: list[str] | None = None
    bias: str = "none"   # "none" | "all" | "lora_only"


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Wrap model with PEFT LoRA adapters and return the PeftModel.

    If config.target_modules is None, attention projection leaf names are
    auto-discovered. Fails loudly if no matching modules are found so you
    don't silently get a zero-adapter run.
    """
    try:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "peft not installed. Run: pip install 'peft>=0.12.0'"
        ) from exc

    target_modules = config.target_modules
    if target_modules is None:
        from .loader import discover_attention_leaf_names, discover_linear_modules
        target_modules = discover_attention_leaf_names(model)
        if not target_modules:
            # Fall back to all linear leaf names
            all_linear = discover_linear_modules(model)
            target_modules = sorted({p.split(".")[-1] for p in all_linear})

    if not target_modules:
        raise RuntimeError(
            "No LoRA target modules found. "
            "Pass explicit config.target_modules or verify the model has nn.Linear layers."
        )

    logger.info("Applying LoRA r=%d α=%d to modules: %s", config.r, config.lora_alpha, target_modules)

    peft_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias=config.bias,
        inference_mode=False,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def save_lora(model: nn.Module, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(output_dir))
        logger.info("Saved PEFT adapter → %s", output_dir)
    except AttributeError:
        import torch
        torch.save(model.state_dict(), output_dir / "adapter_model.pt")
        logger.info("Saved full state_dict → %s/adapter_model.pt", output_dir)


def load_lora(model: nn.Module, adapter_dir: str | Path) -> nn.Module:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("peft not installed.") from exc

    adapter_dir = Path(adapter_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    logger.info("Loaded LoRA adapter ← %s", adapter_dir)
    return model
