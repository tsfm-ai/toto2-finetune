from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..model.lora import LoRAConfig


@dataclass
class TrainConfig:
    # Window geometry
    context_length: int = 672       # 28 days @ 1h
    prediction_length: int = 24     # day-ahead
    stride: int = 24                # non-overlapping windows per default
    min_context_frac: float = 0.5   # require at least half a context window

    # Optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # Training loop
    num_steps: int = 1000
    batch_size: int = 32
    log_every: int = 50
    eval_every: int = 200

    # Fine-tune mode
    finetune_mode: str = "lora"     # "lora" | "full" | "head"
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Device / precision
    device: str | None = None       # None → auto (MPS on M4, CUDA if present)
    dtype: str = "float32"          # "float32" | "bfloat16"

    seed: int = 42

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "bfloat16": torch.bfloat16}[self.dtype]
