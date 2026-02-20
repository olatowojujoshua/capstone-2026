# scripts/eval_slices.py
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from tqdm import tqdm

# ----------------------------
# Path / project imports
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # capstone_2026/
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, MODELS_DIR  # noqa: E402


# ----------------------------
# Metrics helpers (sklearn-free)
# ----------------------------
def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def p90_abs_err(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.quantile(np.abs(y_true - y_pred), 0.90))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "p90_abs_error": p90_abs_err(y_true, y_pred),
        "mean_y_true": float(np.mean(y_true)),
        "mean_y_pred": float(np.mean(y_pred)),
        "n": int(len(y_true)),
    }


def residual_variance_by_decile(df: pd.DataFrame, y_col: str, pred_col: str) -> pd.DataFrame:
    resid = df[y_col] - df[pred_col]
    decile = pd.qcut(df[pred_col], q=10, duplicates="drop")
    out = (
        pd.DataFrame({"pred_decile": decile, "resid": resid})
        .groupby("pred_decile", observed=True)["resid"]
        .agg(resid_var="var", resid_std="std", resid_mean="mean", n="size")
        .reset_index()
    )
    return out


def plot_resid_variance(decile_tbl: pd.DataFrame, out_fp: Path) -> None:
    def _fmt_interval(val) -> str:
        if hasattr(val, "left") and hasattr(val, "right"):
            return f"({val.left:.1f}, {val.right:.1f}]"
        return str(val)

    labels = [_fmt_interval(x) for x in decile_tbl["pred_decile"]]
    plt.figure(figsize=(7, 4))
    plt.plot(decile_tbl.index, decile_tbl["resid_std"], marker="o")
    plt.xticks(decile_tbl.index, labels, rotation=30, ha="right")
    plt.xlabel("Predicted fare decile")
    plt.ylabel("Residual std dev")
    plt.title("Residual variability by prediction decile")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


# ----------------------------
# Slice definitions
# ----------------------------
def add_trip_length_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buckets based on trip_miles:
      short: 0-2
      medium: >2-6
      long: >6
    """
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


def add_hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    # prefer pickup_datetime if present else time_bucket
    if "pickup_datetime" in df.columns:
        ts = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df["hour"] = ts.dt.hour
    elif "time_bucket" in df.columns:
        ts = pd.to_datetime(df["time_bucket"], errors="coerce")
        df["hour"] = ts.dt.hour
    else:
        df["hour"] = np.nan
    return df


def top_n_zones(df: pd.DataFrame, n: int = 20) -> set:
    if "PULocationID" not in df.columns:
        return set()
    vc = df["PULocationID"].value_counts().head(n)
    return set(vc.index.tolist())


def slice_table(df: pd.DataFrame, group_col: str, y_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    # observed=True reduces category cartesian explosion if col is categorical
    for key, g in df.groupby(group_col, observed=True):
        g2 = g.dropna(subset=[y_col, pred_col])
        if len(g2) == 0:
            continue
        m = compute_metrics(g2[y_col].values, g2[pred_col].values)
        m[group_col] = key
        rows.append(m)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("n", ascending=False).reset_index(drop=True)
    return out


# ----------------------------
# Main
# ----------------------------
DEFAULT_MODELS = [
    {"name": "baseline_hgb", "dir": "models/baseline_hgb_model", "file": "baseline_hgb_model.joblib"},
    {"name": "model_log", "dir": "models/model_log", "file": "baseline_hgb_model.joblib"},
    {"name": "model_segmented", "dir": "models/model_segmented", "file": "baseline_hgb_model.joblib"},
    {"name": "model_quantile", "dir": "models/model_quantile", "file": "baseline_hgb_model.joblib"},
    {"name": "linear_regression", "dir": "models/linear_regression_model", "file": "linear_regression_model.joblib"},
    {"name": "model_rf", "dir": "models/model_rf", "file": "random_forest_model.joblib"},
    {"name": "model_xgb", "dir": "models/model_xgb", "file": "xgboost_model.joblib"},
    {"name": "model_gbr", "dir": "models/model_gbr", "file": "gradient_boosting_model.joblib"},
]


def parse_model_specs(specs: list[str]) -> list[dict]:
    models = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) == 1:
            models.append({"name": parts[0], "dir": parts[0], "file": "baseline_hgb_model.joblib"})
        elif len(parts) == 2:
            models.append({"name": parts[0], "dir": parts[1], "file": "baseline_hgb_model.joblib"})
        else:
            models.append({"name": parts[0], "dir": parts[1], "file": parts[2]})
    return models


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Single model folder containing model + features.json",
    )
    p.add_argument("--model_file", type=str, default=None)
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model specs: name:dir[:file]",
    )
    p.add_argument("--features_file", type=str, default="features.json")
    p.add_argument("--model_table_dir", type=str, default="data/processed/model_table")
    p.add_argument("--test_month", type=str, default="2021-10", help="YYYY-MM parquet exists in model_table_dir")
    p.add_argument("--target", type=str, default="base_passenger_fare", help="Target column name")
    p.add_argument("--sample_n", type=int, default=0, help="0 = no sampling, else sample N rows from test month")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--top_zones_n", type=int, default=20)
    p.add_argument("--hetero", action="store_true", help="write heteroscedasticity diagnostics")
    p.add_argument("--segment", type=str, default=None, choices=["short", "medium", "long"], help="optional trip length segment filter")
    args = p.parse_args()

    mt_dir = Path(args.model_table_dir)
    if not mt_dir.is_absolute():
        mt_dir = PROJECT_ROOT / mt_dir

    test_fp = mt_dir / f"{args.test_month}.parquet"
    if not test_fp.exists():
        raise FileNotFoundError(f"Missing test parquet: {test_fp}")

    out_dir = PROJECT_ROOT / "reports" / "slices"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        models = parse_model_specs(args.models)
    elif args.model_dir:
        model_file = args.model_file or "baseline_hgb_model.joblib"
        models = [{"name": Path(args.model_dir).name, "dir": args.model_dir, "file": model_file}]
    else:
        models = DEFAULT_MODELS

    overall_rows = []
    for model_spec in models:
        model_dir = Path(model_spec["dir"])
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir

        model_path = model_dir / model_spec["file"]
        feats_path = model_dir / args.features_file
        metrics_path = model_dir / "metrics.json"
        if not feats_path.exists():
            feats_path = model_dir.parent / args.features_file
        if not metrics_path.exists():
            metrics_path = model_dir.parent / "metrics.json"

        if not model_path.exists():
            print(f"[WARN] Missing model file: {model_path}")
            continue

        model = load(model_path)
        with open(feats_path, "r", encoding="utf-8") as f:
            feats_json = json.load(f)

        target_transform = "none"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_json = json.load(f)
            if isinstance(metrics_json, dict):
                target_transform = metrics_json.get("target_transform", "none")

        # Accept either {"features":[...]} or {"numeric":[...], "categorical":[...], "all":[...]}
        if isinstance(feats_json, dict):
            if "features" in feats_json:
                feature_cols = feats_json["features"]
            elif "all" in feats_json:
                feature_cols = feats_json["all"]
            else:
                feature_cols = []
                for k in ["numeric", "categorical"]:
                    if k in feats_json and isinstance(feats_json[k], list):
                        feature_cols.extend(feats_json[k])
        elif isinstance(feats_json, list):
            feature_cols = feats_json
        else:
            raise ValueError("features.json format not recognized.")

        if not feature_cols:
            raise ValueError("No feature columns found in features.json")

        needed_cols = set(
            feature_cols
            + [args.target, "trip_miles", "PULocationID", "hvfhs_license_num", "pickup_datetime", "time_bucket"]
        )
        df = pd.read_parquet(test_fp, columns=[c for c in needed_cols if c is not None])

        if args.sample_n and args.sample_n > 0 and len(df) > args.sample_n:
            df = df.sample(n=args.sample_n, random_state=args.random_state).reset_index(drop=True)

        if "pickup_datetime" in df.columns:
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        if "time_bucket" in df.columns:
            df["time_bucket"] = pd.to_datetime(df["time_bucket"], errors="coerce")

        df = add_trip_length_bucket(df)
        df = add_hour_of_day(df)

        if args.segment:
            df = df[df["trip_len_bucket"] == args.segment].copy()

        topz = top_n_zones(df, n=args.top_zones_n)
        if "PULocationID" in df.columns and len(topz) > 0:
            df["pu_zone_topN"] = df["PULocationID"].apply(lambda z: f"top_{args.top_zones_n}" if z in topz else "other")
        else:
            df["pu_zone_topN"] = "unknown"

        y = df[args.target].astype(float).values
        X = df.reindex(columns=feature_cols)

        preds = model.predict(X)
        if target_transform == "log1p":
            preds = np.expm1(preds)
        df["y_pred"] = preds

        model_name = model_spec["name"]
        overall = compute_metrics(y, preds)
        overall.update(
            {
                "test_month": args.test_month,
                "sample_n": int(len(df)),
                "model_name": model_name,
                "model_path": str(model_path),
            }
        )
        overall_rows.append(overall)

        if args.hetero:
            hetero_tbl = residual_variance_by_decile(df, args.target, "y_pred")
            hetero_csv = out_dir / f"hetero_resid_by_decile_{model_name}_{args.test_month}.csv"
            hetero_tbl.to_csv(hetero_csv, index=False)
            hetero_plot = out_dir / f"hetero_resid_by_decile_{model_name}_{args.test_month}.png"
            plot_resid_variance(hetero_tbl, hetero_plot)
            print(f"[OK] Wrote {hetero_csv}")
            print(f"[OK] Wrote {hetero_plot}")

        slice_specs = [
            ("trip_len_bucket", "trip_len_bucket"),
            ("pu_zone_topN", "pu_zone_topN"),
            ("PULocationID", "PULocationID"),
            ("platform", "hvfhs_license_num"),
            ("hour", "hour"),
        ]

        for name, col in slice_specs:
            if col not in df.columns:
                continue
            tbl = slice_table(df, col, args.target, "y_pred")
            out_csv = out_dir / f"slices_{name}_{model_name}_{args.test_month}.csv"
            tbl.to_csv(out_csv, index=False)
            print(f"[OK] Wrote {out_csv} | rows: {len(tbl):,}")

    if overall_rows:
        overall_df = pd.DataFrame(overall_rows)
        overall_df = overall_df.sort_values("rmse", ascending=True).reset_index(drop=True)
        overall_csv = out_dir / f"overall_metrics_{args.test_month}.csv"
        overall_df.to_csv(overall_csv, index=False)
        summary_json = out_dir / f"overall_summary_{args.test_month}.json"
        summary = {
            "test_month": args.test_month,
            "best_rmse": overall_df.iloc[0].to_dict(),
            "best_mae": overall_df.sort_values("mae").iloc[0].to_dict(),
            "best_r2": overall_df.sort_values("r2", ascending=False).iloc[0].to_dict(),
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Wrote {overall_csv}")
        print(f"[OK] Wrote {summary_json}")

    print("\n[OK] Slice evaluation complete.")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
