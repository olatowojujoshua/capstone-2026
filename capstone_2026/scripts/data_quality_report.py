# scripts/data_quality_report.py
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- project imports ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add the folder that contains `src/` to the path
# If your structure is PROJECT_ROOT/capstone_2026/src, this will work:
CAPSTONE_ROOT = PROJECT_ROOT / "capstone_2026"
if CAPSTONE_ROOT.exists():
    sys.path.insert(0, str(CAPSTONE_ROOT))
else:
    # fallback: maybe src/ is at project root
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DIR, PROCESSED_DIR

# ---- parquet schema (metadata only) ----
try:
    import pyarrow.parquet as pq
except ImportError as e:
    raise ImportError(
        "pyarrow is required for memory-safe schema reads. Install with: pip install pyarrow"
    ) from e

# Columns expected in HVFHS (safe if missing)
CORE_COLS = [
    "hvfhs_license_num",
    "request_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "sales_tax",
    "bcf",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    "shared_request_flag",
    "wav_request_flag",
    "wav_match_flag",
]

CRITICAL_COLS = [
    "pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
]

NUMERIC_COLS = [
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tips",
    "driver_pay",
    "tolls",
    "sales_tax",
    "bcf",
    "congestion_surcharge",
    "airport_fee",
]


def safe_to_datetime(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def to_numeric_safe(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def quantile_safe(s: pd.Series, q: float):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    return float(s.quantile(q))


def get_parquet_columns(fp: Path):
    """
    Memory-safe: reads only parquet metadata (no data load).
    """
    try:
        return set(pq.ParquetFile(fp).schema.names)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        type=str,
        default="fhvhv_tripdata_*.parquet",
        help="Filename glob pattern in data/raw/",
    )
    parser.add_argument(
        "--pickup_delay_max_sec",
        type=int,
        default=7200,
        help="Max allowed pickup delay in seconds (default 2 hours).",
    )
    parser.add_argument(
        "--winsor_lo",
        type=float,
        default=0.005,
        help="Lower winsor quantile (default 0.5%).",
    )
    parser.add_argument(
        "--winsor_hi",
        type=float,
       default=0.995,
        help="Upper winsor quantile (default 99.5%).",
    )
    args = parser.parse_args()

    raw_dir = Path(RAW_DIR)
    files = sorted(raw_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {raw_dir} with pattern {args.pattern}")

    out_dir = Path(PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "data_quality_report.csv"

    rows = []

    for fp in tqdm(files, desc="Data quality report (per month)"):
        month_tag = fp.stem.replace("fhvhv_tripdata_", "")

        available = get_parquet_columns(fp)
        read_cols = [c for c in CORE_COLS if (available is None or c in available)]

        df = pd.read_parquet(fp, columns=read_cols)

        raw_rows = len(df)
        present_cols = df.columns.tolist()

        # --- stage 1: datetime conversion ---
        for c in ["request_datetime", "pickup_datetime", "dropoff_datetime"]:
            safe_to_datetime(df, c)

        # --- stage 2: drop missing critical ---
        crit = [c for c in CRITICAL_COLS if c in df.columns]
        before_dropna = len(df)
        df = df.dropna(subset=crit)
        dropped_missing_critical = before_dropna - len(df)

        # --- stage 3: numeric conversion ---
        for c in NUMERIC_COLS:
            to_numeric_safe(df, c)

        # --- stage 4: invalid/ impossible trips ---
        removed_trip_miles = 0
        if "trip_miles" in df.columns:
            bad = ~(df["trip_miles"] > 0)
            removed_trip_miles = int(bad.sum())
            df = df.loc[~bad]

        removed_trip_time = 0
        if "trip_time" in df.columns:
            bad = ~(df["trip_time"] > 0)
            removed_trip_time = int(bad.sum())
            df = df.loc[~bad]

        removed_base_fare = 0
        if "base_passenger_fare" in df.columns:
            bad = ~(df["base_passenger_fare"] > 0)
            removed_base_fare = int(bad.sum())
            df = df.loc[~bad]

        # --- stage 5: pickup delay proxy filter ---
        pickup_delay_out_of_range = 0
        pickup_delay_missing = np.nan
        pickup_delay_mean = np.nan

        if "request_datetime" in df.columns and "pickup_datetime" in df.columns:
            delay = (df["pickup_datetime"] - df["request_datetime"]).dt.total_seconds()
            pickup_delay_missing = int(delay.isna().sum())
            pickup_delay_mean = float(delay.dropna().mean()) if len(delay.dropna()) else np.nan

            bad_delay = delay.notna() & ((delay < 0) | (delay > args.pickup_delay_max_sec))
            pickup_delay_out_of_range = int(bad_delay.sum())
            df = df.loc[~bad_delay]

        # --- stage 6: fare_per_mile + winsor thresholds (report only) ---
        fare_per_mile_lo = fare_per_mile_hi = np.nan
        trip_miles_lo = trip_miles_hi = np.nan
        trip_time_lo = trip_time_hi = np.nan
        base_fare_lo = base_fare_hi = np.nan

        if "base_passenger_fare" in df.columns and "trip_miles" in df.columns:
            df["fare_per_mile"] = df["base_passenger_fare"] / df["trip_miles"]

        if "trip_miles" in df.columns:
            trip_miles_lo = quantile_safe(df["trip_miles"], args.winsor_lo)
            trip_miles_hi = quantile_safe(df["trip_miles"], args.winsor_hi)
        if "trip_time" in df.columns:
            trip_time_lo = quantile_safe(df["trip_time"], args.winsor_lo)
            trip_time_hi = quantile_safe(df["trip_time"], args.winsor_hi)
        if "base_passenger_fare" in df.columns:
            base_fare_lo = quantile_safe(df["base_passenger_fare"], args.winsor_lo)
            base_fare_hi = quantile_safe(df["base_passenger_fare"], args.winsor_hi)
        if "fare_per_mile" in df.columns:
            fare_per_mile_lo = quantile_safe(df["fare_per_mile"], args.winsor_lo)
            fare_per_mile_hi = quantile_safe(df["fare_per_mile"], args.winsor_hi)

        final_rows = len(df)
        total_removed = raw_rows - final_rows
        pct_removed = (total_removed / raw_rows) * 100 if raw_rows else np.nan

        rows.append(
            {
                "month": month_tag,
                "raw_rows": raw_rows,
                "columns_present": len(present_cols),
                "dropped_missing_critical": dropped_missing_critical,
                "removed_trip_miles_nonpositive": removed_trip_miles,
                "removed_trip_time_nonpositive": removed_trip_time,
                "removed_base_fare_nonpositive": removed_base_fare,
                "removed_pickup_delay_out_of_range": pickup_delay_out_of_range,
                "pickup_delay_missing_count": pickup_delay_missing,
                "pickup_delay_mean_sec": pickup_delay_mean,
                "winsor_lo_q": args.winsor_lo,
                "winsor_hi_q": args.winsor_hi,
                "trip_miles_lo": trip_miles_lo,
                "trip_miles_hi": trip_miles_hi,
                "trip_time_lo": trip_time_lo,
                "trip_time_hi": trip_time_hi,
                "base_fare_lo": base_fare_lo,
                "base_fare_hi": base_fare_hi,
                "fare_per_mile_lo": fare_per_mile_lo,
                "fare_per_mile_hi": fare_per_mile_hi,
                "final_rows_after_filters": final_rows,
                "total_removed": total_removed,
                "pct_removed": pct_removed,
            }
        )

    report = pd.DataFrame(rows).sort_values("month")
    report.to_csv(out_csv, index=False)
    print(f"\n[OK] Data quality report saved -> {out_csv}")
    print("\nPreview:")
    print(report.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
