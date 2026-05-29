from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

TOTO2_MODEL_IDS = {
    "22M":  "Datadog/Toto-2.0-22M",
    "313M": "Datadog/Toto-2.0-313M",
    "1B":   "Datadog/Toto-2.0-1B",
    "2.5B": "Datadog/Toto-2.0-2.5B",
}


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_toto2(
    model_id: str,
    revision: str | None = None,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """
    Load a Toto2Model from HuggingFace hub.

    model_id: HF repo id or shorthand key from TOTO2_MODEL_IDS
              e.g. "313M" → "Datadog/Toto-2.0-313M"

    Requires toto2 to be installed:
      pip install "git+https://github.com/DataDog/toto.git#subdirectory=toto2"
    """
    try:
        from toto2 import Toto2Model
    except ImportError as exc:
        raise ImportError(
            "toto2 package not found. Install with:\n"
            '  pip install "git+https://github.com/DataDog/toto.git#subdirectory=toto2"'
        ) from exc

    model_id = TOTO2_MODEL_IDS.get(model_id, model_id)

    if device is None:
        device = _auto_device()

    logger.info("Loading %s → device=%s dtype=%s", model_id, device, dtype)
    model: nn.Module = Toto2Model.from_pretrained(model_id, revision=revision)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def discover_linear_modules(model: nn.Module) -> list[str]:
    """Full dot-paths for every nn.Linear in the model (useful for inspection)."""
    return [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]


def discover_attention_leaf_names(model: nn.Module) -> list[str]:
    """
    Unique leaf names (last path segment) of attention-related Linear layers.
    PEFT matches target_modules by leaf name when applied to custom models.
    """
    attention_keywords = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv", "out_proj"}
    seen: set[str] = set()
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in attention_keywords:
                seen.add(leaf)
    return sorted(seen)
