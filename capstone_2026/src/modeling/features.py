import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def load_features_and_target() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load per-month model tables and sample for training.
    Returns X (features) and y (target = base_passenger_fare).
    """
    from src.config import PROCESSED_DIR

    model_table_dir = Path(PROCESSED_DIR) / "model_table"
    files = sorted(model_table_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No model tables found in {model_table_dir}")

    # Load and sample 5% for quick baseline training
    parts = []
    for fp in files:
        df = pd.read_parquet(fp)
        parts.append(df.sample(frac=0.05, random_state=42))
    df = pd.concat(parts, ignore_index=True)

    # Simple feature set
    feature_cols = [
        "PULocationID", "DOLocationID", "trip_miles", "trip_time",
        "pickup_delay_sec", "fare_per_mile"
    ]
    # Add zone-time features if present
    for col in ["trip_count", "avg_pickup_delay_sec", "med_fare_per_mile"]:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].copy()
    y = df["base_passenger_fare"].copy()

    # Basic imputation
    X = X.fillna(X.median(numeric_only=True))
    return X, y
