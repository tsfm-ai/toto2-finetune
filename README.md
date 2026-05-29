# toto2-finetune

[![CI](https://github.com/tsfm-ai/toto2-finetune/actions/workflows/ci.yml/badge.svg)](https://github.com/tsfm-ai/toto2-finetune/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Fine-tuning and evaluation harness for [Toto 2](https://github.com/DataDog/toto) on energy time series.**

Toto 2 is Datadog's state-of-the-art time-series foundation model — a decoder-only patched transformer with alternating time and variate attention, trained on trillions of observations. Official fine-tuning support is not yet public. This repo builds the missing training stack: LoRA and full fine-tuning via the low-level `forward()` interface, a DataFrame-first data pipeline with past/future covariate handling, walk-forward backtesting, and a leakage audit framework designed for energy forecasting.

The goal is to find out whether Toto 2 fine-tuning beats Chronos 2 LoRA on domain-specific energy data — and to build the harness that makes that comparison clean.

---

## Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.4 (MPS on Apple Silicon, CUDA on GPU)
- [toto2](https://github.com/DataDog/toto) — installed separately (see below)
- [peft](https://github.com/huggingface/peft) ≥ 0.12

## Install

```bash
# 1. Clone and install toto2-finetune
git clone https://github.com/tsfm-ai/toto2-finetune
cd toto2-finetune
pip install -e ".[dev]"

# 2. Install Datadog's toto2 (pulls gluonts, dd-unit-scaling, jaxtyping)
pip install "git+https://github.com/DataDog/toto.git#subdirectory=toto2"
```

---

## Quickstart — zero-shot inference

```python
import pandas as pd
from toto2ft.data.schema import DataSchema
from toto2ft.inference.pipeline import Toto2Pipeline

pipe = Toto2Pipeline.from_pretrained("Datadog/Toto-2.0-313M")

# Long-format DataFrame: one row per timestamp per group
df = pd.read_parquet("nyiso_load.parquet")

schema = DataSchema(
    timestamp_col="timestamp",
    group_cols=["iso", "zone"],
    target_cols=["load_mw"],
    future_covariate_cols=["temperature_forecast", "hour_sin", "hour_cos", "is_holiday"],
    frequency="1h",
)

# q_preds: np.ndarray (Q=9, N_windows, n_targets, horizon)
q_preds = pipe.predict(df, schema, context_length=672, prediction_length=24)

# Quantile levels: [0.1, 0.2, ..., 0.9]
median = q_preds[4]   # P50
```

---

## Fine-tuning

Toto 2's `forward()` is lower-level than `forecast()` — it returns predictions in the model's normalized space alongside the `loc` and `scale` parameters needed to invert the transform. This repo handles that correctly: **loss is computed in normalized space, inference output is denormalized to original units**.

### Target-only LoRA (Gate 2)

```python
import torch
from torch.utils.data import DataLoader
from toto2ft.data.schema import DataSchema
from toto2ft.data.windowing import WindowDataset, collate_windows
from toto2ft.model.loader import load_toto2
from toto2ft.training.configs import TrainConfig
from toto2ft.training.trainer import train
from toto2ft.training.checkpointing import save_checkpoint

schema = DataSchema(
    timestamp_col="timestamp",
    group_cols=["iso", "zone"],
    target_cols=["load_mw"],
    frequency="1h",
)

config = TrainConfig(
    context_length=672,     # must be a multiple of patch_size (32)
    prediction_length=24,   # padded to 32 internally — one patch
    finetune_mode="lora",
    num_steps=1000,
    batch_size=32,
    learning_rate=1e-5,
)

train_ds = WindowDataset(train_df, schema, config.context_length, config.prediction_length)
val_ds   = WindowDataset(val_df,   schema, config.context_length, config.prediction_length)

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  collate_fn=collate_windows)
val_loader   = DataLoader(val_ds,   batch_size=64,                shuffle=False, collate_fn=collate_windows)

model = load_toto2("Datadog/Toto-2.0-313M")
log = train(model, train_loader, config, val_loader=val_loader)

save_checkpoint(model, "./runs/nyiso_313m_lora", step=config.num_steps, log=log)
```

### Multivariate LoRA with covariates (Gate 3)

```python
schema = DataSchema(
    timestamp_col="timestamp",
    group_cols=["iso", "zone"],
    target_cols=["load_mw"],
    # Past-observed: available in context, not future horizon
    past_covariate_cols=[
        "temperature_actual",
        "wind_generation_actual",
        "solar_generation_actual",
    ],
    # Known-future: forecast quantities available at forecast creation time
    future_covariate_cols=[
        "temperature_forecast",
        "wind_generation_forecast",
        "solar_generation_forecast",
        "hour_sin", "hour_cos", "day_of_week", "is_holiday",
    ],
    availability_col="availability_time",   # for leakage audit
    frequency="1h",
)
```

The CPM mask is built automatically: context visible, future targets hidden, future known covariates visible. Loss is computed on target variates only.

### Reload and predict

```python
from toto2ft.inference.pipeline import Toto2Pipeline

pipe = Toto2Pipeline.from_checkpoint(
    base_model_id="Datadog/Toto-2.0-313M",
    checkpoint_dir="./runs/nyiso_313m_lora",
)

q_preds = pipe.predict(test_df, schema, context_length=672, prediction_length=24)
# q_preds: (9, N, 1, 24) — denormalized to original MW scale
```

---

## What we had to figure out

Toto 2's `forward()` is undocumented for training use. Three non-obvious constraints:

**1. Patch alignment.** The total sequence length `(context + horizon)` must be divisible by `patch_size = 32`. `context_length` must be a multiple of 32. The horizon is right-padded to the next patch boundary before every `forward()` call. `num_return_steps` is in **patches**, not timesteps — pass `ceil(H / 32)`.

```
H=24 → H_pad=32 (1 patch) → num_return_steps=1
Output: (Q=9, B, V, 1, 32) → flatten → slice [:24]
```

**2. LoRA target modules.** Toto 2 uses a fused `in_proj` for QKV (not split `q_proj`/`k_proj`/`v_proj`). The correct PEFT targets are `["in_proj", "out_proj"]`. Auto-discovery is built in — it inspects the live model rather than assuming an architecture.

**3. Normalized output space.** `forward()` returns predictions in the model's internal normalized space. `outputs.loc` and `outputs.scale` are the per-timestep inverse-transform parameters. Training loss is computed in normalized space; inference output is denormalized: `q_real = q_norm * scale + loc`.

---

## Covariate schema

| Type | When available | CPM treatment | In loss |
|---|---|---|---|
| `target_cols` | Context only (future hidden) | Context visible, future **hidden** | ✓ future only |
| `past_covariate_cols` | Context only | Context visible, future **hidden** | ✗ |
| `future_covariate_cols` | Full window (known at forecast time) | Context visible, future **visible** | ✗ |

The leakage audit (`toto2ft.data.leakage`) checks that every feature's `availability_time` ≤ the forecast creation time. For real energy forecasting — where weather forecasts, ISO forecasts, and outage data all have distinct availability timestamps — this is non-negotiable.

---

## Evaluation

Walk-forward backtesting with stratified metrics:

```python
from toto2ft.evaluation.backtest import walk_forward_backtest
from toto2ft.evaluation.slices import stratified_metrics, horizon_step_metrics

metrics, actuals, q_preds = walk_forward_backtest(
    model=model,
    df=test_df,
    schema=schema,
    context_length=672,
    prediction_length=24,
    device=torch.device("mps"),
)
# metrics: MAE, RMSE, sMAPE, MASE, WQL, P50_bias,
#          P10/P90 empirical CDF, P10_P90_coverage, interval_sharpness

# Stratify by zone, season, holiday, hour bucket...
slice_metrics = stratified_metrics(actuals, q_preds, metadata_df)

# Per-horizon-step error curve
step_metrics = horizon_step_metrics(actuals, q_preds)
```

---

## Experiment gates

The harness is structured around four decision gates. YAML configs live in `experiments/`.

| Gate | Model | Mode | Pass condition |
|---|---|---|---|
| **1 — Plumbing** | 22M | full FT | Loss decreases, no NaN, checkpoint save/reload works |
| **2 — Adaptation** | 313M | LoRA | ≥3–5% WQL/MAE improvement over Toto 2 zero-shot |
| **3 — Covariates** | 313M | LoRA | Beats Chronos 2 LoRA (same covariates) by ≥2–3% globally or ≥5% on stress slices |
| **4 — Scale** | 1B | LoRA | Materially beats 313M after cost normalization |

Gate 1 status: **✓ passing** — training stack runs end-to-end on CPU/MPS, LoRA attaches correctly, checkpoint save/reload verified, inference produces valid denormalized quantiles.

Run the Gate 1 check:

```bash
python examples/quickstart_target_only.py
```

---

## Model sizes

| Shorthand | HuggingFace ID | Params | Use |
|---|---|---|---|
| `"22M"` | `Datadog/Toto-2.0-22M` | 22M | Plumbing / fast iteration |
| `"313M"` | `Datadog/Toto-2.0-313M` | 313M | Primary fine-tuning target |
| `"1B"` | `Datadog/Toto-2.0-1B` | 1B | Scale test after 313M proves lift |
| `"2.5B"` | `Datadog/Toto-2.0-2.5B` | 2.5B | Final benchmark only |

---

## API reference

### `DataSchema`

| Field | Description |
|---|---|
| `timestamp_col` | Datetime column name |
| `group_cols` | Series identifier columns (e.g. `["iso", "zone"]`) |
| `target_cols` | Target variate(s) — loss computed here only |
| `past_covariate_cols` | Context-only covariates (not visible in future) |
| `future_covariate_cols` | Known-future covariates (visible across full window) |
| `availability_col` | Column with feature availability timestamp (leakage audit) |
| `frequency` | Pandas frequency string (e.g. `"1h"`) |

### `TrainConfig`

| Field | Default | Description |
|---|---|---|
| `context_length` | 672 | Context window; must be a multiple of `patch_size` |
| `prediction_length` | 24 | Forecast horizon; padded to next patch boundary internally |
| `patch_size` | 32 | Toto 2 patch size — do not change unless model differs |
| `finetune_mode` | `"lora"` | `"lora"` / `"full"` / `"head"` |
| `learning_rate` | `1e-5` | AdamW learning rate |
| `num_steps` | 1000 | Total training steps |
| `batch_size` | 32 | Training batch size |
| `warmup_steps` | 100 | Linear LR warmup steps |
| `grad_clip` | 1.0 | Gradient norm clip |
| `device` | auto | `"cpu"` / `"mps"` / `"cuda"` |
| `dtype` | `"float32"` | `"float32"` / `"bfloat16"` |

### `LoRAConfig`

| Field | Default | Description |
|---|---|---|
| `r` | 8 | LoRA rank |
| `lora_alpha` | 16 | Scaling factor |
| `lora_dropout` | 0.05 | Dropout on LoRA path |
| `target_modules` | auto | Leaf module names; `None` → auto-discover from live model |
| `bias` | `"none"` | `"none"` / `"all"` / `"lora_only"` |

### `Toto2Pipeline`

| Method | Description |
|---|---|
| `from_pretrained(model_id, ...)` | Zero-shot pipeline from HuggingFace |
| `from_checkpoint(base_model_id, checkpoint_dir, ...)` | Fine-tuned pipeline; loads base + adapter |
| `predict(df, schema, context_length, prediction_length, ...)` | Returns `(Q=9, N, n_targets, H)` denormalized quantiles |
| `save(output_dir)` | Save adapter/weights |

---

## Development

```bash
git clone https://github.com/tsfm-ai/toto2-finetune
cd toto2-finetune
pip install -e ".[dev]"

# Unit tests — no model download required (run in ~0.1s)
pytest

# Gate 1 end-to-end check — downloads 22M model (~85 MB)
python examples/quickstart_target_only.py

# Multivariate template (requires real data)
python examples/nyiso_multivariate.py --data /path/to/data.parquet
```

Tests cover: pinball loss shape handling, CPM mask construction, rolling window generation, and the full metric suite. No internet or GPU required.

---

## Hosted forecasting

Want to run Toto 2, Chronos 2, and other TSFMs without managing infrastructure? [TSFM.ai](https://tsfm.ai) offers a hosted forecasting API.

---

## License

Apache-2.0. Toto 2 model weights are subject to [Datadog's license](https://github.com/DataDog/toto/blob/main/LICENSE).
