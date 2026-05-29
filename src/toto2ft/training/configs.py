from __future__ import annotations

from dataclasses import dataclass, field

import math

import torch

from ..model.lora import LoRAConfig


@dataclass
class TrainConfig:
    # Window geometry
    # context_length MUST be a multiple of patch_size (32).
    # 672 = 21 × 32 = 28 days @ 1h ✓
    context_length: int = 672
    prediction_length: int = 24     # day-ahead; padded to 32 internally
    stride: int = 24
    min_context_frac: float = 0.5
    patch_size: int = 32            # Toto 2 patch size; override if model differs

    # Optimizer
    # 1e-4 is the empirically validated rate for LoRA on Toto 2 targets
    # (in_proj, out_proj, fc1, fc2). Use 1e-5 for full fine-tuning.
    learning_rate: float = 1e-4
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

    @property
    def prediction_length_padded(self) -> int:
        """Prediction length rounded up to next patch boundary."""
        return math.ceil(self.prediction_length / self.patch_size) * self.patch_size

    @property
    def n_return_patches(self) -> int:
        return self.prediction_length_padded // self.patch_size

    def validate_geometry(self) -> None:
        if self.context_length % self.patch_size != 0:
            raise ValueError(
                f"context_length={self.context_length} must be a multiple of "
                f"patch_size={self.patch_size}. "
                f"Nearest valid values: "
                f"{self.context_length - self.context_length % self.patch_size} or "
                f"{self.context_length + self.patch_size - self.context_length % self.patch_size}"
            )
