# scripts/train_model.py

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



TARGET_COL = "base_passenger_fare"


def load_month_table(fp: Path, sample_n: int | None):
    df = pd.read_parquet(fp)

    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)

    return df


def load_split_tables(model_table_dir: Path, months: list[str], sample_per_month: int | None):
    parts = []
    for m in months:
        fp = model_table_dir / f"{m}.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"Missing model table for month {m}: {fp}")
        parts.append(load_month_table(fp, sample_per_month))

    return pd.concat(parts, ignore_index=True)


def pick_feature_columns(df: pd.DataFrame):
    requested = [
        "PULocationID",
        "DOLocationID",
        "trip_miles",
        "trip_time",
        "pickup_delay_sec",
        "shared_request_flag",
        "wav_request_flag",
        "wav_match_flag",
        "trip_count",
        "avg_pickup_delay_sec",
    ]

    return [c for c in requested if c in df.columns]


def load_feature_list(feature_json: str | None, df: pd.DataFrame) -> list[str]:
    if not feature_json:
        return pick_feature_columns(df)

    with open(feature_json, "r") as f:
        feats = json.load(f)

    return [c for c in feats if c in df.columns]


def add_trip_length_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "trip_miles" not in df.columns:
        df["trip_len_bucket"] = "unknown"
        return df

    miles = df["trip_miles"].astype(float)
    df["trip_len_bucket"] = pd.cut(
        miles,
        bins=[-np.inf, 2, 6, np.inf],
        labels=["short", "medium", "long"],
    ).astype("object")
    return df


def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    abs_err = np.abs(y_true - y_pred)
    p90 = float(np.quantile(abs_err, 0.90))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "p90_abs_error": p90,
        "mean_y_true": float(np.mean(y_true)),
        "mean_y_pred": float(np.mean(y_pred)),
        "n": int(len(y_true)),
    }



# -------------------------
# Training
# -------------------------

def train_pipeline(args):
    model_table_dir = Path(args.model_table_dir)

    # month ranges
    def month_range(start, end):
        ms = pd.period_range(start, end, freq="M")
        return [str(m) for m in ms]

    train_months = month_range(args.train_start, args.train_end)

    print("[INFO] Train months:", train_months)
    print("[INFO] Val month:", args.val_month)
    print("[INFO] Test month:", args.test_month)

    print("\n[LOAD] Training data...")
    df_train = load_split_tables(model_table_dir, train_months, args.sample_per_month)

    print("[LOAD] Validation data...")
    df_val = load_split_tables(model_table_dir, [args.val_month], args.sample_per_month)

    print("[LOAD] Test data...")
    df_test = load_split_tables(model_table_dir, [args.test_month], args.sample_per_month)

    print(f"[SHAPE] train={df_train.shape} val={df_val.shape} test={df_test.shape}")

    df_train = add_trip_length_bucket(df_train)
    df_val = add_trip_length_bucket(df_val)
    df_test = add_trip_length_bucket(df_test)

    feats = load_feature_list(args.features_json, df_train)

    print(f"[INFO] Using {len(feats)} numeric features")

    segments = ["all"]
    if args.segment_by_trip_length:
        segments = ["short", "medium", "long"]

    metrics = {
        "features": feats,
        "train_months": train_months,
        "val_month": args.val_month,
        "test_month": args.test_month,
        "target_transform": "log1p" if args.log_target else "none",
        "segments": {},
    }

    trained_models = {}

    for seg in segments:
        if seg == "all":
            train_seg = df_train
            val_seg = df_val
            test_seg = df_test
        else:
            train_seg = df_train[df_train["trip_len_bucket"] == seg]
            val_seg = df_val[df_val["trip_len_bucket"] == seg]
            test_seg = df_test[df_test["trip_len_bucket"] == seg]

        X_train = train_seg[feats].fillna(0)
        y_train = train_seg[TARGET_COL]

        X_val = val_seg[feats].fillna(0)
        y_val = val_seg[TARGET_COL]

        X_test = test_seg[feats].fillna(0)
        y_test = test_seg[TARGET_COL]

        if args.log_target:
            y_train = np.log1p(y_train)

        if args.model_type == "linear":
            print(f"\n[TRAIN] LinearRegression (segment={seg})...")
            model = LinearRegression()
        elif args.model_type == "gbr":
            print(f"\n[TRAIN] GradientBoostingRegressor (segment={seg})...")
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
            )
        elif args.model_type == "rf":
            print(f"\n[TRAIN] RandomForestRegressor (segment={seg})...")
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        elif args.model_type == "xgb":
            print(f"\n[TRAIN] XGBoostRegressor (segment={seg})...")
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError("xgboost is not installed. Please pip install xgboost") from exc
            model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
        else:
            print(f"\n[TRAIN] HistGradientBoostingRegressor (segment={seg})...")
            model = HistGradientBoostingRegressor(
                max_depth=8,
                learning_rate=0.08,
                max_iter=300,
                random_state=42,
                loss=args.loss,
                quantile=args.quantile if args.loss == "quantile" else None,
            )

        model.fit(X_train, y_train)

        print(f"[EVAL] Validation (segment={seg})...")
        val_pred = model.predict(X_val)
        if args.log_target:
            val_pred = np.expm1(val_pred)
        val_metrics = eval_metrics(y_val, val_pred)

        print(f"[EVAL] Test (segment={seg})...")
        test_pred = model.predict(X_test)
        if args.log_target:
            test_pred = np.expm1(test_pred)
        test_metrics = eval_metrics(y_test, test_pred)

        metrics["segments"][seg] = {
            "validation": val_metrics,
            "test": test_metrics,
            "n_train": int(len(train_seg)),
            "n_val": int(len(val_seg)),
            "n_test": int(len(test_seg)),
        }
        trained_models[seg] = model

    # -------------------------
    # Save
    # -------------------------

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "features.json", "w") as f:
        json.dump(feats, f, indent=2)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    for seg in segments:
        seg_dir = out_dir if seg == "all" else out_dir / f"segment={seg}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        if args.model_type == "linear":
            model_name = "linear_regression_model.joblib"
        elif args.model_type == "gbr":
            model_name = "gradient_boosting_model.joblib"
        elif args.model_type == "rf":
            model_name = "random_forest_model.joblib"
        elif args.model_type == "xgb":
            model_name = "xgboost_model.joblib"
        else:
            model_name = "baseline_hgb_model.joblib"
        model_fp = seg_dir / model_name
        joblib.dump(trained_models[seg], model_fp)

    print("\n[SAVED]")
    print("Dir ->", out_dir)


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_table_dir", type=str, default="data/processed/model_table")

    p.add_argument("--train_start", type=str, default="2021-01")
    p.add_argument("--train_end", type=str, default="2021-08")

    p.add_argument("--val_month", type=str, default="2021-09")
    p.add_argument("--test_month", type=str, default="2021-10")

    p.add_argument("--sample_per_month", type=int, default=500000)

    p.add_argument(
        "--features_json",
        type=str,
        default=None,
        help="Path to features.json to use fixed feature list",
    )

    p.add_argument("--output_dir", type=str, default="models/final_model")
    p.add_argument("--model_type", type=str, default="hgb", choices=["hgb", "linear", "xgb", "rf", "gbr"], help="model type to train")
    p.add_argument("--log_target", action="store_true", help="train on log1p(target) and invert for metrics")
    p.add_argument("--segment_by_trip_length", action="store_true", help="train separate models for short/medium/long trips")
    p.add_argument("--loss", type=str, default="squared_error", choices=["squared_error", "absolute_error", "poisson", "quantile"], help="loss function for HistGradientBoostingRegressor")
    p.add_argument("--quantile", type=float, default=0.5, help="quantile value when loss=quantile (e.g., 0.5 for median)")

    return p.parse_args()


def main():
    args = parse_args()
    train_pipeline(args)


if __name__ == "__main__":
    main()
