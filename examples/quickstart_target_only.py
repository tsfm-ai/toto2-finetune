"""
Gate 1 quickstart: target-only LoRA fine-tuning on synthetic load data.

Proves the training stack runs end-to-end on the 22M model:
  ✓ loss decreases (typically 70-80% drop over 600 steps)
  ✓ no NaNs in quantile output
  ✓ checkpoint saves and reloads
  ✓ inference wrapper produces valid denormalized quantiles

Key findings baked in:
  - LoRA target_modules: in_proj + out_proj (attention) + fc1 + fc2 (FFN)
    → ~393K trainable params vs full 22M
  - lr=1e-4: 10x higher than typical — needed because LoRA adapters start at zero
  - Loss computed in normalized space (forward() returns normalized predictions)
  - Harder synthetic data (3 zones, diurnal + weekly + spikes) exercises the model

Usage:
  python examples/quickstart_target_only.py
"""
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

from toto2ft.data.schema import DataSchema
from toto2ft.data.windowing import WindowDataset, collate_windows
from toto2ft.inference.pipeline import Toto2Pipeline
from toto2ft.model.lora import LoRAConfig
from toto2ft.model.loader import load_toto2
from toto2ft.training.checkpointing import save_checkpoint
from toto2ft.training.configs import TrainConfig
from toto2ft.training.trainer import train

# ── Synthetic hourly load — 3 zones, diurnal + weekly seasonality + noise spikes
rng = np.random.default_rng(42)
n = 3000  # ~125 days
timestamps = pd.date_range("2023-01-01", periods=n, freq="1h")
t = np.arange(n)

records = []
for zone, base, amp in [("A", 1500, 300), ("B", 2200, 450), ("C", 900, 200)]:
    # Diurnal + weekly seasonality
    load = (
        base
        + amp * np.sin(t * 2 * np.pi / 24)
        + 0.3 * amp * np.sin(t * 2 * np.pi / (24 * 7))
        + rng.normal(0, base * 0.02, n)
    )
    # Sparse demand spikes (simulates events / cold snaps)
    spike_idx = rng.choice(n, size=n // 50, replace=False)
    load[spike_idx] += rng.uniform(0.1 * amp, 0.4 * amp, len(spike_idx))
    for i, ts in enumerate(timestamps):
        records.append({"timestamp": ts, "iso": "NYISO", "zone": zone, "load": float(load[i])})

df = pd.DataFrame(records)

schema = DataSchema(
    timestamp_col="timestamp",
    group_cols=["iso", "zone"],
    target_cols=["load"],
    frequency="1h",
)

split = int(0.8 * n)
train_df = df[df["timestamp"] < timestamps[split]].copy()
val_df   = df[df["timestamp"] >= timestamps[split - 128]].copy()

# ── Config ────────────────────────────────────────────────────────────────────
# context_length=96  (= 3 × 32, patch-aligned ✓)
# prediction_length=24 (padded to 32 = 1 patch internally)
# LoRA targets: in_proj + out_proj (attention) + fc1 + fc2 (FFN) → ~393K params
# lr=1e-4: empirically validated — 1e-5 is too low for LoRA cold-start
config = TrainConfig(
    context_length=96,
    prediction_length=24,
    finetune_mode="lora",
    lora=LoRAConfig(
        r=8,
        lora_alpha=16,
        target_modules=["in_proj", "out_proj", "fc1", "fc2"],
    ),
    num_steps=600,
    batch_size=8,
    learning_rate=1e-4,
    warmup_steps=60,
    log_every=50,
    eval_every=200,
    device="cpu",  # CPU for plumbing run; set "mps" or "cuda" for real training
)

# ── Datasets ──────────────────────────────────────────────────────────────────
train_ds = WindowDataset(train_df, schema, config.context_length, config.prediction_length)
val_ds   = WindowDataset(val_df,   schema, config.context_length, config.prediction_length)

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  collate_fn=collate_windows)
val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, collate_fn=collate_windows)

print(f"Train windows: {len(train_ds)}  Val windows: {len(val_ds)}")
print(f"C={config.context_length}  H={config.prediction_length}  "
      f"H_pad={config.prediction_length_padded}  "
      f"total={config.context_length + config.prediction_length_padded} "
      f"(= {(config.context_length + config.prediction_length_padded)//32} patches)")

# ── Load model ────────────────────────────────────────────────────────────────
model = load_toto2("Datadog/Toto-2.0-22M", device="cpu")

# ── Train ─────────────────────────────────────────────────────────────────────
log = train(model, train_loader, config, val_loader=val_loader)

first_loss = log[0]["train_loss"]
final_loss = log[-1]["train_loss"]
pct_drop = (first_loss - final_loss) / first_loss * 100
print(f"\nFirst loss : {first_loss:.4f}")
print(f"Final loss : {final_loss:.4f}")
print(f"Loss drop  : {pct_drop:.1f}%")

# A healthy LoRA run drops 50-80% — if it's flat, check lr and target_modules
assert final_loss < first_loss * 0.80, (
    f"Loss barely moved ({pct_drop:.1f}% drop). "
    "Expected ≥20% — check learning_rate and target_modules."
)

# ── Save & reload ─────────────────────────────────────────────────────────────
save_checkpoint(model, "./runs/quickstart_22m", step=config.num_steps, log=log)

pipeline = Toto2Pipeline.from_checkpoint(
    base_model_id="Datadog/Toto-2.0-22M",
    checkpoint_dir="./runs/quickstart_22m",
    device="cpu",
)

# ── Inference sanity check ────────────────────────────────────────────────────
q_preds = pipeline.predict(val_df, schema, config.context_length, config.prediction_length)
assert not np.isnan(q_preds).any(),  "NaN in quantile predictions"
assert not np.isinf(q_preds).any(), "Inf in quantile predictions"
assert (np.diff(q_preds, axis=0) >= -1e-4).all(), "Quantiles not monotone"

print(f"\nq_preds shape : {q_preds.shape}  (Q=9, N, n_target=1, H=24)")
print(f"P50 mean      : {q_preds[4].mean():.1f}  MW")
print(f"P10 mean      : {q_preds[0].mean():.1f}  MW")
print(f"P90 mean      : {q_preds[8].mean():.1f}  MW")
print(f"Monotone      : {(np.diff(q_preds, axis=0) >= -1e-4).all()}")

print("\n✓ Gate 1 PASS: LoRA learns, checkpoint saves/reloads, inference is valid.")
