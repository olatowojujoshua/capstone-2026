import argparse
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INTERIM_DIR, PROCESSED_DIR, TIME_BUCKET_MINUTES


def make_time_bucket(ts: pd.Series) -> pd.Series:
    return ts.dt.floor(f"{TIME_BUCKET_MINUTES}min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="*_clean.parquet")
    args = parser.parse_args()

    interim_dir = Path(INTERIM_DIR)
    zt_fp = Path(PROCESSED_DIR) / "zone_time_features" / "zone_time_features.parquet"
    out_dir = Path(PROCESSED_DIR) / "model_table"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not zt_fp.exists():
        raise FileNotFoundError(f"Missing zone_time_features: {zt_fp}")

    # Load zone-time features once
    zt = pd.read_parquet(zt_fp)
    zt["time_bucket"] = pd.to_datetime(zt["time_bucket"], errors="coerce")

    files = sorted(interim_dir.glob(args.pattern))
    print(f"[INFO] Found {len(files)} files with pattern '{args.pattern}' in {interim_dir}")
    if not files:
        print("[ERROR] No cleaned parquet files found.")
        raise FileNotFoundError("No cleaned parquet files found.")

    # Only read columns we need (saves RAM)
    needed_trip_cols = [
        "hvfhs_license_num",
        "pickup_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_miles",
        "trip_time",
        "base_passenger_fare",
        "tips",
        "driver_pay",
        "fare_per_mile",
        "pickup_delay_sec",
        "shared_request_flag",
        "wav_request_flag",
        "wav_match_flag",
    ]

    for fp in tqdm(files, desc="Building model_table per month"):
        month = fp.stem.replace("_clean", "")  # e.g., 2021-01

        # Read only available columns
        df = pd.read_parquet(fp)
        cols = [c for c in needed_trip_cols if c in df.columns]
        df = df[cols].copy()

        # Ensure datetime + bucket
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df["time_bucket"] = make_time_bucket(df["pickup_datetime"])
        df["month"] = month

        # Filter zone-time to same month
        zt_m = zt[zt["month"] == month].copy()
        if len(zt_m) == 0:
            print(f"[WARN] No zone-time features for {month}. Writing without join.")
            out_fp = out_dir / f"{month}.parquet"
            df.to_parquet(out_fp, index=False, compression="snappy")
            continue

        # Merge (left join)
        df_out = df.merge(
            zt_m[["PULocationID", "time_bucket", "trip_count", "avg_pickup_delay_sec", "med_fare_per_mile"]],
            on=["PULocationID", "time_bucket"],
            how="left",
        )

        out_fp = out_dir / f"{month}.parquet"
        df_out.to_parquet(out_fp, index=False, compression="snappy")
        print(f"[OK] wrote {out_fp.name} | rows: {len(df_out):,}")

    print(f"\n[DONE] Saved per-month model tables in: {out_dir}")


if __name__ == "__main__":
    main()
