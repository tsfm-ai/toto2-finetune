"""
Gate 2: 313M LoRA vs. Toto 2 zero-shot.

Runs the full comparison on synthetic (default) or real data:
  1. Zero-shot baseline: walk-forward backtest with pretrained 313M
  2. LoRA fine-tuning: 2000 steps, lr=1e-4, in_proj+out_proj+fc1+fc2
  3. Fine-tuned backtest: same walk-forward protocol
  4. % delta on WQL, MAE, RMSE, sMAPE, P10/P90 coverage

Pass condition (Gate 2): ≥3% WQL or MAE improvement over zero-shot.

Usage:
  # Synthetic data (no downloads needed beyond the 313M weights, ~1.2 GB):
  python examples/gate2_313m_comparison.py

  # Real data:
  python examples/gate2_313m_comparison.py --data /path/to/load.parquet \\
      --target-col load_mw --group-cols iso zone

  # Faster smoke test on 22M:
  python examples/gate2_313m_comparison.py --model-id Datadog/Toto-2.0-22M \\
      --num-steps 400
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

import toto2ft  # noqa: F401 — loads compat shim before toto2

from toto2ft.data.schema import DataSchema
from toto2ft.data.windowing import WindowDataset, collate_windows
from toto2ft.evaluation.compare import compare_zero_shot_vs_ft, print_comparison_table
from toto2ft.model.lora import LoRAConfig
from toto2ft.model.loader import load_toto2
from toto2ft.training.checkpointing import save_checkpoint
from toto2ft.training.configs import TrainConfig
from toto2ft.training.trainer import train

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Gate 2: 313M zero-shot vs LoRA comparison")
parser.add_argument("--data", default=None, help="Parquet path; omit for synthetic data")
parser.add_argument("--target-col", default="load", help="Target column name")
parser.add_argument("--group-cols", nargs="+", default=["iso", "zone"])
parser.add_argument("--model-id", default="Datadog/Toto-2.0-313M")
parser.add_argument("--num-steps", type=int, default=2000)
parser.add_argument("--output-dir", default="./runs/gate2_ft")
parser.add_argument("--context-length", type=int, default=672)
parser.add_argument("--device", default=None, help="cpu | mps | cuda (auto-detected)")
args = parser.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────
def make_synthetic_df(n: int = 5000) -> pd.DataFrame:
    """3-zone synthetic load: diurnal + weekly seasonality + demand spikes."""
    rng = np.random.default_rng(42)
    t = np.arange(n)
    timestamps = pd.date_range("2022-01-01", periods=n, freq="1h")
    records = []
    for zone, base, amp in [("A", 1500, 300), ("B", 2200, 450), ("C", 900, 200)]:
        load = (
            base
            + amp * np.sin(t * 2 * np.pi / 24)
            + 0.3 * amp * np.sin(t * 2 * np.pi / (24 * 7))
            + rng.normal(0, base * 0.02, n)
        )
        spike_idx = rng.choice(n, size=n // 30, replace=False)
        load[spike_idx] += rng.uniform(0.1 * amp, 0.5 * amp, len(spike_idx))
        for i, ts in enumerate(timestamps):
            records.append({"timestamp": ts, "iso": "NYISO", "zone": zone,
                            "load": float(load[i])})
    return pd.DataFrame(records)


if args.data:
    print(f"Loading data from {args.data}")
    df = pd.read_parquet(args.data)
    target_col = args.target_col
    group_cols = args.group_cols
else:
    print("No --data provided — using synthetic 3-zone load data")
    df = make_synthetic_df(n=5000)
    target_col = "load"
    group_cols = ["iso", "zone"]

schema = DataSchema(
    timestamp_col="timestamp",
    group_cols=group_cols,
    target_cols=[target_col],
    frequency="1h",
)

# Chronological split: 70% train, 15% val, 15% test
all_ts = df["timestamp"].sort_values().unique()
n_ts = len(all_ts)
train_cut = all_ts[int(0.70 * n_ts)]
val_cut   = all_ts[int(0.85 * n_ts)]

train_df = df[df["timestamp"] < train_cut].copy()
val_df   = df[(df["timestamp"] >= pd.Timestamp(train_cut) - pd.Timedelta(hours=args.context_length))
              & (df["timestamp"] < val_cut)].copy()
test_df  = df[df["timestamp"] >= val_cut].copy()

print(f"\nSplit:  train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,} rows")
print(f"Groups: {df[group_cols].drop_duplicates().shape[0]}")


# ── Config ───────────────────────────────────────────────────────────────────
device_str = args.device or (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
device = torch.device(device_str)
print(f"Device: {device}\n")

config = TrainConfig(
    context_length=args.context_length,
    prediction_length=24,
    finetune_mode="lora",
    lora=LoRAConfig(
        r=8,
        lora_alpha=16,
        target_modules=["in_proj", "out_proj", "fc1", "fc2"],
    ),
    num_steps=args.num_steps,
    batch_size=32 if device_str != "cpu" else 8,
    learning_rate=1e-4,
    warmup_steps=max(50, args.num_steps // 10),
    early_stopping_patience=5,
    log_every=100,
    eval_every=200,
    device=device_str,
)

train_ds = WindowDataset(train_df, schema, config.context_length, config.prediction_length)
val_ds   = WindowDataset(val_df,   schema, config.context_length, config.prediction_length,
                         stride=config.prediction_length)

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                          collate_fn=collate_windows)
val_loader   = DataLoader(val_ds,   batch_size=max(8, config.batch_size * 2),
                          shuffle=False, collate_fn=collate_windows)

print(f"Train windows: {len(train_ds):,}  Val windows: {len(val_ds):,}")


# ── Fine-tune ─────────────────────────────────────────────────────────────────
print(f"\n── Fine-tuning {args.model_id} ──────────────────────────────────────────")
ft_model = load_toto2(args.model_id, device="cpu")  # load to CPU, trainer moves to device
log = train(ft_model, train_loader, config, val_loader=val_loader)

first_loss = log[0]["train_loss"]
final_loss = log[-1]["train_loss"]
print(f"\nTrain loss:  {first_loss:.4f} → {final_loss:.4f}  "
      f"({(first_loss - final_loss) / first_loss * 100:.1f}% drop)")

val_entries = [e for e in log if "val_loss" in e]
if val_entries:
    print(f"Val loss:    {val_entries[0]['val_loss']:.4f} → {val_entries[-1]['val_loss']:.4f}")

save_checkpoint(ft_model, args.output_dir, step=len(log), log=log)
print(f"Checkpoint → {args.output_dir}")


# ── Compare ──────────────────────────────────────────────────────────────────
print(f"\n── Zero-shot vs Fine-tuned on test split ────────────────────────────────")
results = compare_zero_shot_vs_ft(
    ft_model=ft_model,
    base_model_id=args.model_id,
    test_df=test_df,
    schema=schema,
    context_length=config.context_length,
    prediction_length=config.prediction_length,
    device=device,
    batch_size=max(8, config.batch_size * 2),
)

print_comparison_table(results)

# ── Gate 2 verdict ───────────────────────────────────────────────────────────
dp = results["delta_pct"]
wql_ok = dp.get("WQL", 0) <= -3.0
mae_ok = dp.get("MAE", 0) <= -3.0

print("\n── Gate 2 verdict ───────────────────────────────────────────────────────")
if wql_ok or mae_ok:
    print(f"✓ PASS  WQL Δ={dp.get('WQL', 0):+.1f}%  MAE Δ={dp.get('MAE', 0):+.1f}%")
    print("  Fine-tuning beats zero-shot by ≥3% on at least one primary metric.")
else:
    print(f"✗ FAIL  WQL Δ={dp.get('WQL', 0):+.1f}%  MAE Δ={dp.get('MAE', 0):+.1f}%")
    print("  Did not reach the ≥3% threshold on WQL or MAE.")
    print("  Consider: more steps, larger r, or real domain-specific data.")
