import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "capstone_2026"
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INTERIM_DIR, PROCESSED_DIR, TIME_BUCKET_MINUTES

def make_time_bucket(ts: pd.Series) -> pd.Series:
    return ts.dt.floor(f"{TIME_BUCKET_MINUTES}min")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="*_clean.parquet")
    args = parser.parse_args()

    out_dir = Path(PROCESSED_DIR) / "zone_time_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(INTERIM_DIR).glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No cleaned files found in {INTERIM_DIR} with pattern {args.pattern}")

    all_parts = []
    for fp in tqdm(files, desc="Building zone-time features"):
        df = pd.read_parquet(fp, columns=[
            "PULocationID", "pickup_datetime", "pickup_delay_sec",
            "fare_per_mile", "base_passenger_fare"
        ])

        df["time_bucket"] = make_time_bucket(df["pickup_datetime"])

        agg = df.groupby(["PULocationID", "time_bucket"], observed=True).agg(
            trip_count=("base_passenger_fare", "size"),
            avg_pickup_delay_sec=("pickup_delay_sec", "mean"),
            med_fare_per_mile=("fare_per_mile", "median"),
        ).reset_index()

        # month tag for partitioning
        month_tag = fp.stem.replace("_clean", "")
        agg["month"] = month_tag
        all_parts.append(agg)

    features = pd.concat(all_parts, ignore_index=True)
    out_fp = out_dir / "zone_time_features.parquet"
    features.to_parquet(out_fp, index=False, compression="snappy")
    print(f"[OK] Feature store saved -> {out_fp} | rows: {len(features):,}")

if __name__ == "__main__":
    main()
