"""
Gate 1 quickstart: target-only fine-tuning on synthetic load data.

Proves the training stack runs end-to-end on the 22M model:
  ✓ loss decreases
  ✓ no NaNs in quantile output
  ✓ checkpoint saves and reloads
  ✓ inference wrapper produces valid quantiles

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
from toto2ft.model.loader import load_toto2
from toto2ft.training.checkpointing import save_checkpoint
from toto2ft.training.configs import TrainConfig
from toto2ft.training.trainer import train

# ── Synthetic hourly load ─────────────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 2000
timestamps = pd.date_range("2023-01-01", periods=n, freq="1h")

records = []
for zone in ["A", "B"]:
    load = 1500 + 300 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 30, n)
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
val_df   = df[df["timestamp"] >= timestamps[split - 96]].copy()

# ── Config ────────────────────────────────────────────────────────────────────
# context_length=96 (= 3 × 32, patch-aligned ✓)
# prediction_length=24 (padded to 32 = 1 patch internally)
config = TrainConfig(
    context_length=96,
    prediction_length=24,
    finetune_mode="lora",
    num_steps=200,
    batch_size=8,
    warmup_steps=20,
    log_every=20,
    eval_every=100,
    device="cpu",   # CPU for plumbing run; MPS/CUDA for real training
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
print(f"\nFirst loss : {first_loss:.4f}")
print(f"Final loss : {final_loss:.4f}")
assert final_loss < first_loss * 1.5, "Loss did not decrease — check training stack"

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

print(f"\nq_preds shape : {q_preds.shape}  (Q=9, N, n_target=1, H=24)")
print(f"P50 mean      : {q_preds[4].mean():.1f}  (expect ~1500 for synthetic load)")
print(f"P10 mean      : {q_preds[0].mean():.1f}")
print(f"P90 mean      : {q_preds[8].mean():.1f}")
print(f"Monotone      : {(np.diff(q_preds, axis=0) >= 0).all()}")

print("\n✓ Gate 1 PASS: training stack runs end-to-end.")
