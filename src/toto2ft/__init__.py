# Compat shim must be first — patches F.scaled_dot_product_attention
# before toto2 is imported so enable_gqa works on PyTorch < 2.5.
from . import compat as _compat  # noqa: F401

from .model.loader import load_toto2
from .model.losses import pinball_loss
from .model.lora import LoRAConfig, apply_lora, save_lora, load_lora
from .training.configs import TrainConfig
from .training.trainer import train
from .training.checkpointing import save_checkpoint, load_checkpoint
from .inference.pipeline import Toto2Pipeline

__version__ = "0.1.0"

__all__ = [
    "load_toto2",
    "pinball_loss",
    "LoRAConfig",
    "apply_lora",
    "save_lora",
    "load_lora",
    "TrainConfig",
    "train",
    "save_checkpoint",
    "load_checkpoint",
    "Toto2Pipeline",
]
