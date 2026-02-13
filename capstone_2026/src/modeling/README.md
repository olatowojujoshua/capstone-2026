# Modeling

This module provides a baseline fare-prediction pipeline for NYC HVFHS data.

## Files

- `src/modeling/features.py`: Load and sample model tables; construct feature matrix.
- `src/modeling/train.py`: Train a Gradient Boosting regressor and compute evaluation metrics.
- `scripts/train_model.py`: CLI entry point to train and version a model.

## Quick start

```bash
# Ensure model tables exist (see scripts/build_model_table.py)
python scripts/train_model.py --output-dir baseline_gbt
```

Outputs:
- Model: `models/baseline_gbt/fare_gbt_model.joblib`
- Metrics: `models/baseline_gbt/metrics.csv`

## Features

Uses trip-level and zone-time features:
- Trip miles, time, fare per mile, pickup delay
- Zone-time aggregates (trip count, avg pickup delay, median fare/mile)
- Flags (shared request, WAV match)

## Model

GradientBoostingRegressor:
- 200 trees, lr=0.05, max_depth=4
- Evaluated on 20% hold-out (MAE, RMSE, R²)

## Extending

- Add new features in `features.py`
- Swap models in `train.py`
- Save feature list alongside model for reproducible inference.
