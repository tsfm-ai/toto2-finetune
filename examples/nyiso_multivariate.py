"""
Gate 3 template: multivariate LoRA fine-tuning on NYISO load with covariates.

Expects a DataFrame at DATA_PATH with columns:
  timestamp, iso, zone, load_mw,
  temperature_actual, wind_generation_actual, solar_generation_actual,
  temperature_forecast, wind_generation_forecast, solar_generation_forecast,
  hour_sin, hour_cos, day_of_week, is_holiday

Usage:
  python examples/nyiso_multivariate.py --data /path/to/nyiso.parquet
"""
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

from toto2ft.data.leakage import audit_availability
from toto2ft.data.schema import DataSchema
from toto2ft.data.windowing import WindowDataset, collate_windows
from toto2ft.evaluation.backtest import walk_forward_backtest
from toto2ft.model.loader import load_toto2
from toto2ft.training.checkpointing import save_checkpoint
from toto2ft.training.configs import TrainConfig
from toto2ft.training.trainer import train

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--model-id", default="Datadog/Toto-2.0-313M")
parser.add_argument("--output-dir", default="./runs/nyiso_multivariate_313m_lora")
parser.add_argument("--num-steps", type=int, default=2000)
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(args.data)

PAST_COVARIATES = [
    "temperature_actual",
    "wind_generation_actual",
    "solar_generation_actual",
]
FUTURE_COVARIATES = [
    "temperature_forecast",
    "wind_generation_forecast",
    "solar_generation_forecast",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_holiday",
]

schema = DataSchema(
    timestamp_col="timestamp",
    group_cols=["iso", "zone"],
    target_cols=["load_mw"],
    past_covariate_cols=PAST_COVARIATES,
    future_covariate_cols=FUTURE_COVARIATES,
    availability_col="availability_time" if "availability_time" in df.columns else None,
    frequency="1h",
)
schema.validate(df)

# ── Leakage audit ─────────────────────────────────────────────────────────────
if schema.availability_col:
    violations = audit_availability(
        df, schema.all_value_cols, schema.availability_col, "forecast_creation_time"
    )
    if len(violations):
        raise RuntimeError(f"Leakage detected: {len(violations)} rows. Fix before training.")
    print("Leakage audit PASSED.")

# ── Train / val / test splits ─────────────────────────────────────────────────
train_df = df[df["timestamp"] < "2024-01-01"].copy()
val_df   = df[(df["timestamp"] >= "2024-01-01") & (df["timestamp"] < "2024-07-01")].copy()
test_df  = df[df["timestamp"] >= "2024-07-01"].copy()

# ── Config ───────────────────────────────────────────────────────────────────
config = TrainConfig(
    context_length=672,
    prediction_length=24,
    finetune_mode="lora",
    num_steps=args.num_steps,
    batch_size=32,
    warmup_steps=200,
    log_every=50,
    eval_every=200,
)

# ── Datasets ──────────────────────────────────────────────────────────────────
train_ds = WindowDataset(train_df, schema, config.context_length, config.prediction_length)
val_ds   = WindowDataset(val_df, schema, config.context_length, config.prediction_length, stride=24)

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  collate_fn=collate_windows)
val_loader   = DataLoader(val_ds,   batch_size=64,                shuffle=False, collate_fn=collate_windows)

print(f"Train: {len(train_ds)} windows  Val: {len(val_ds)} windows  Test: {len(test_df)} rows")

# ── Model ─────────────────────────────────────────────────────────────────────
model = load_toto2(args.model_id)

# ── Train ─────────────────────────────────────────────────────────────────────
log = train(model, train_loader, config, val_loader=val_loader)

save_checkpoint(model, args.output_dir, step=args.num_steps, log=log)
print(f"\nCheckpoint → {args.output_dir}")

# ── Walk-forward backtest on test split ───────────────────────────────────────
device = next(model.parameters()).device
metrics, _, _ = walk_forward_backtest(
    model=model,
    df=test_df,
    schema=schema,
    context_length=config.context_length,
    prediction_length=config.prediction_length,
    device=device,
    stride=24,
)

print("\n── Test metrics ──────────────────────────────────────────────────────")
for k, v in sorted(metrics.items()):
    print(f"  {k:<25s} {v:.4f}")
