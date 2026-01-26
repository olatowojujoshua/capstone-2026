import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

# --- make imports work when running scripts/ ---
PROJECT_ROOT = Path(__file__).resolve().parents[1] / "capstone_2026"
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INTERIM_DIR, MODELS_DIR




# Helpers

def month_range(start: str, end: str) -> List[str]:
    """Inclusive YYYY-MM month range."""
    s = pd.Period(start, freq="M")
    e = pd.Period(end, freq="M")
    if e < s:
        raise ValueError("end must be >= start")
    return [str(p) for p in pd.period_range(s, e, freq="M")]

def load_month_clean(month: str, columns: List[str]) -> pd.DataFrame:
    fp = Path(INTERIM_DIR) / f"{month}_clean.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing cleaned file: {fp}")
    return pd.read_parquet(fp, columns=[c for c in columns if c is not None])

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # We prefer request_datetime if present; fallback to pickup_datetime
    ts_col = "request_datetime" if "request_datetime" in df.columns else "pickup_datetime"
    if ts_col not in df.columns:
        raise ValueError("Need request_datetime or pickup_datetime for time features")

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.copy()
    df["hour"] = ts.dt.hour.astype("Int16")
    df["dow"] = ts.dt.dayofweek.astype("Int16")  # 0=Mon
    df["month_num"] = ts.dt.month.astype("Int16")
    return df

def sample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n <= 0:
        return df
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)

def p90_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.percentile(np.abs(y_true - y_pred), 90))

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape(y_true, y_pred),
        "P90_abs_error": p90_abs_error(y_true, y_pred),
        "n": int(len(y_true)),
    }



# Main training routine

def build_dataset(months: List[str], per_month_sample: int, cols_needed: List[str]) -> pd.DataFrame:
    parts = []
    for m in months:
        df = load_month_clean(m, columns=cols_needed)
        df = add_time_features(df)

        # Keep only needed modeling columns (drop extras to reduce RAM)
        keep = [c for c in cols_needed if c in df.columns]
        keep += ["hour", "dow", "month_num"]
        keep = list(dict.fromkeys(keep))  
        df = df[keep]

        df = sample_df(df, per_month_sample, seed=42)
        df["data_month"] = m  
        parts.append(df)

        print(f"[LOAD] {m}: sampled {len(df):,} rows")

    out = pd.concat(parts, ignore_index=True)
    print(f"[DATASET] total rows: {len(out):,}")
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_start", type=str, default="2021-01")
    parser.add_argument("--train_end", type=str, default="2021-08")
    parser.add_argument("--val_month", type=str, default="2021-09")
    parser.add_argument("--test_month", type=str, default="2021-10")
    parser.add_argument("--per_month_sample", type=int, default=600_000,
                        help="Rows sampled per month for training/val/test (controls RAM).")
    parser.add_argument("--target", type=str, default="base_passenger_fare")
    parser.add_argument("--model_name", type=str, default="baseline_hgb_dev")
    args = parser.parse_args()



    # Baseline fare prediction features 
    # NOTE: We do NOT use zone_time_features here. This is baseline pricing.
    base_cols = [
        "hvfhs_license_num",
        "PULocationID",
        "DOLocationID",
        "trip_miles",
        "trip_time",
        "request_datetime",   
        "pickup_datetime",   
        args.target,
    ]

    train_months = month_range(args.train_start, args.train_end)
    val_months = [args.val_month]
    test_months = [args.test_month]

    print("\n=== BUILD TRAIN ===")
    train_df = build_dataset(train_months, args.per_month_sample, base_cols)

    print("\n=== BUILD VAL ===")
    val_df = build_dataset(val_months, max(150_000, args.per_month_sample // 4), base_cols)

    print("\n=== BUILD TEST ===")
    test_df = build_dataset(test_months, max(150_000, args.per_month_sample // 4), base_cols)

    # Define features/target
    target = args.target
    feature_cols = [
        "trip_miles", "trip_time", "hour", "dow", "month_num",
        "PULocationID", "DOLocationID", "hvfhs_license_num",
    ]

    # Drop rows with missing target
    train_df = train_df.dropna(subset=[target])
    val_df = val_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])

    X_train, y_train = train_df[feature_cols], train_df[target].astype(float)
    X_val, y_val = val_df[feature_cols], val_df[target].astype(float)
    X_test, y_test = test_df[feature_cols], test_df[target].astype(float)

    # Convert categorical features to string type (ensure they're not numeric)
    for col in ["PULocationID", "DOLocationID", "hvfhs_license_num"]:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Preprocess: numeric + categorical
    numeric_features = ["trip_miles", "trip_time", "hour", "dow", "month_num"]
    categorical_features = ["PULocationID", "DOLocationID", "hvfhs_license_num"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical_features),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_depth=10,
        max_iter=100,
        min_samples_leaf=50,
        random_state=42,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

    print("\n=== TRAIN MODEL (HGB) ===")
    pipe.fit(X_train, y_train)

    print("\n=== EVALUATE ===")
    y_val_pred = pipe.predict(X_val)
    y_test_pred = pipe.predict(X_test)

    metrics = {
        "split": {
            "train_months": train_months,
            "val_month": args.val_month,
            "test_month": args.test_month,
            "per_month_sample_train": int(args.per_month_sample),
            "per_month_sample_val_test": int(max(150_000, args.per_month_sample // 4)),
        },
        "validation": compute_metrics(y_val.to_numpy(), y_val_pred),
        "test": compute_metrics(y_test.to_numpy(), y_test_pred),
        "model": "HistGradientBoostingRegressor",
        "target": target,
        "features": feature_cols,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(MODELS_DIR) / f"{args.model_name}.joblib"
    metrics_path = Path(MODELS_DIR) / f"{args.model_name}_metrics.json"
    schema_path = Path(MODELS_DIR) / "baseline_feature_schema.json"

    dump(pipe, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    schema = {
        "target": target,
        "features": feature_cols,
        "numeric": numeric_features,
        "categorical": categorical_features,
        "time_features_source": "request_datetime (fallback pickup_datetime)",
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"\n[SAVED] Model -> {model_path}")
    print(f"[SAVED] Metrics -> {metrics_path}")
    print(f"[SAVED] Schema -> {schema_path}")

    print("\n=== QUICK RESULTS ===")
    print("Validation:", metrics["validation"])
    print("Test:", metrics["test"])

if __name__ == "__main__":
    main()
