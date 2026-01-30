import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


sys.path.insert(0, str(Path(__file__).parent.parent / "capstone_2026"))

from src.config import RAW_DIR, INTERIM_DIR
from src.cleaning import clean_month_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="fhvhv_tripdata_*.parquet",
                        help="Filename glob pattern in data/raw/")
    args = parser.parse_args()

    files = sorted(Path(RAW_DIR).glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {RAW_DIR} with pattern {args.pattern}")

    for fp in tqdm(files, desc="Cleaning monthly parquet"):
        month_tag = fp.stem.replace("fhvhv_tripdata_", "")
        out_fp = Path(INTERIM_DIR) / f"{month_tag}_clean.parquet"

        df = pd.read_parquet(fp)
        df_clean = clean_month_df(df)

        df_clean.to_parquet(out_fp, index=False, compression="snappy")
        print(f"[OK] {fp.name} -> {out_fp.name} | rows: {len(df_clean):,}")

if __name__ == "__main__":
    main()
