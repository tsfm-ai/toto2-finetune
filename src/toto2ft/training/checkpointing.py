from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    output_dir: str | Path,
    step: int,
    config_dict: dict | None = None,
    log: list[dict] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PEFT models expose save_pretrained; plain nn.Module falls back to state_dict
    try:
        model.save_pretrained(str(output_dir))
    except AttributeError:
        torch.save(model.state_dict(), output_dir / "model.pt")

    meta: dict = {"step": step}
    if config_dict:
        meta["config"] = config_dict
    if log:
        meta["log"] = log

    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Checkpoint saved at step %d → %s", step, output_dir)


def load_checkpoint(
    model: nn.Module,
    checkpoint_dir: str | Path,
) -> tuple[nn.Module, dict]:
    checkpoint_dir = Path(checkpoint_dir)

    # Try PEFT first, then plain state_dict
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(checkpoint_dir), is_trainable=False)
        logger.info("Loaded PEFT checkpoint ← %s", checkpoint_dir)
    except Exception:
        pt_path = checkpoint_dir / "model.pt"
        if pt_path.exists():
            state = torch.load(pt_path, map_location="cpu")
            model.load_state_dict(state)
            logger.info("Loaded state_dict ← %s", pt_path)

    meta_path = checkpoint_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    return model, meta
