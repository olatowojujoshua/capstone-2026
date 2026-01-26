import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "capstone_2026"
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INTERIM_DIR, PROCESSED_DIR, TIME_BUCKET_MINUTES


def pick_timestamp_col(df: pd.DataFrame) -> str:
    # Prefer request_datetime; fallback to pickup_datetime
    if "request_datetime" in df.columns:
        return "request_datetime"
    if "pickup_datetime" in df.columns:
        return "pickup_datetime"
    raise ValueError("No request_datetime or pickup_datetime column found.")


def pick_fare_col(df: pd.DataFrame) -> str:
    # Prefer base_passenger_fare (your baseline target)
    # If you later want total fare, switch here.
    if "base_passenger_fare" in df.columns:
        return "base_passenger_fare"
    # common TLC field name
    if "base_passenger_fare" in df.columns:
        return "base_passenger_fare"
    raise ValueError("No supported fare column found. Expected base_passenger_fare.")


def bucket_time(ts: pd.Series, minutes: int) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    return ts.dt.floor(f"{minutes}min")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0, None) and not pd.isna(b) else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="*_clean.parquet",
                        help="Filename glob pattern inside data/interim/")
    parser.add_argument("--minutes", type=int, default=TIME_BUCKET_MINUTES,
                        help="Time bucket size in minutes (default from src.config)")
    parser.add_argument("--min_trips_per_bucket", type=int, default=200,
                        help="Filter buckets with too few trips to reduce noise")
    args = parser.parse_args()

    interim_dir = Path(INTERIM_DIR)
    files = sorted(interim_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {interim_dir} matching {args.pattern}")

    rows = []

    for fp in tqdm(files, desc="Stability: processing months"):
        df = pd.read_parquet(fp)

        ts_col = pick_timestamp_col(df)
        fare_col = pick_fare_col(df)

        # Minimal columns to reduce memory
        keep = [ts_col, fare_col]
        if "hvfhs_license_num" in df.columns:
            keep.append("hvfhs_license_num")
        if "PULocationID" in df.columns:
            keep.append("PULocationID")

        df = df[keep].copy()

        # Clean
        df[fare_col] = pd.to_numeric(df[fare_col], errors="coerce")
        df = df.dropna(subset=[ts_col, fare_col])
        df = df[df[fare_col] >= 0]

        # Time bucket
        df["time_bucket"] = bucket_time(df[ts_col], args.minutes)
        df = df.dropna(subset=["time_bucket"])

        # Month label from filename (e.g., 2021-06_clean.parquet)
        month_tag = fp.stem.replace("_clean", "")
        df["data_month"] = month_tag

        # Aggregate by month + time_bucket (global), and also by platform if available
        group_cols = ["data_month", "time_bucket"]
        
        # Global aggregation using named aggregation
        agg = df.groupby(group_cols, as_index=False).agg(
            trips=pd.NamedAgg(column="time_bucket", aggfunc="size"),
            mean_fare=pd.NamedAgg(column=fare_col, aggfunc="mean"),
            median_fare=pd.NamedAgg(column=fare_col, aggfunc="median"),
        )
        
        # Add percentiles
        percentiles = df.groupby(group_cols)[fare_col].apply(
            lambda x: pd.Series({
                "p10_fare": np.percentile(x, 10),
                "p90_fare": np.percentile(x, 90)
            })
        ).reset_index()
        agg = agg.merge(percentiles, on=group_cols, how="left")
        agg["level"] = "global"

        # Skip platform-level aggregation due to data quality issues
        agg_p = None

        # Filter buckets with too few trips
        agg = agg[agg["trips"] >= args.min_trips_per_bucket].copy()
        if agg_p is not None:
            agg_p = agg_p[agg_p["trips"] >= args.min_trips_per_bucket].copy()

        def add_volatility_metrics(a: pd.DataFrame, extra_keys=None) -> pd.DataFrame:
            sort_cols = ["data_month", "time_bucket"]
            if extra_keys:
                sort_cols = ["data_month"] + extra_keys + ["time_bucket"]
            a = a.sort_values(sort_cols).copy()

            # previous bucket mean fare
            a["prev_mean_fare"] = a.groupby(["data_month"] + (extra_keys or []))["mean_fare"].shift(1)

            a["abs_jump"] = (a["mean_fare"] - a["prev_mean_fare"]).abs()
            a["pct_jump"] = a.apply(lambda r: safe_div(r["abs_jump"], r["prev_mean_fare"]), axis=1)

            # Flags for “shock” events (you can tune threshold later)
            a["shock_25pct"] = a["pct_jump"] >= 0.25
            a["shock_50pct"] = a["pct_jump"] >= 0.50
            return a

        agg = add_volatility_metrics(agg, extra_keys=None)
        if agg_p is not None:
            agg_p = add_volatility_metrics(agg_p, extra_keys=["hvfhs_license_num"])

        rows.append(agg)
        if agg_p is not None:
            rows.append(agg_p)

        print(f"[OK] {fp.name} -> buckets(global)={len(agg):,} "
              f"{' + buckets(platform)=' + str(len(agg_p)) if agg_p is not None else ''}")

    out = pd.concat(rows, ignore_index=True)

    # Summary metrics per month & level
    summary = (
        out.groupby(["data_month", "level"], as_index=False)
           .agg(
               buckets=("time_bucket", "size"),
               avg_abs_jump=("abs_jump", "mean"),
               p90_abs_jump=("abs_jump", lambda x: float(np.percentile(x.dropna(), 90)) if len(x.dropna()) else np.nan),
               avg_pct_jump=("pct_jump", "mean"),
               p90_pct_jump=("pct_jump", lambda x: float(np.percentile(x.dropna(), 90)) if len(x.dropna()) else np.nan),
               shock_rate_25pct=("shock_25pct", "mean"),
               shock_rate_50pct=("shock_50pct", "mean"),
           )
    )

    # Save both detailed + summary
    out_dir = Path(PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed_fp = out_dir / "price_stability_buckets.parquet"
    summary_fp = out_dir / "price_stability_metrics.parquet"

    out.to_parquet(detailed_fp, index=False, compression="snappy")
    summary.to_parquet(summary_fp, index=False, compression="snappy")

    print("\n[SAVED]")
    print("Detailed buckets ->", detailed_fp)
    print("Summary metrics  ->", summary_fp)
    print("\n=== QUICK VIEW (summary head) ===")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
