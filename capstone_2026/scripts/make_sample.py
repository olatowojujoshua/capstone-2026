import argparse
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "capstone_2026"
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INTERIM_DIR, SAMPLES_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=str, default=None, help="e.g., 2021-06 (optional)")
    parser.add_argument("--n", type=int, default=300_000, help="rows per month sample")
    parser.add_argument("--all_months", action="store_true", help="generate samples for all cleaned months")
    args = parser.parse_args()

    files = sorted(Path(INTERIM_DIR).glob("*_clean.parquet"))
    if not files:
        raise FileNotFoundError(f"No cleaned parquet files found in {INTERIM_DIR}")

    def make_one(fp: Path, n: int):
        month_tag = fp.stem.replace("_clean", "")  # e.g., 2021-06
        out_dir = Path(SAMPLES_DIR) / f"month={month_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(fp)
        sample = df.sample(n=min(n, len(df)), random_state=42)
        out_fp = out_dir / "demo_sample.parquet"
        sample.to_parquet(out_fp, index=False, compression="snappy")
        print(f"[OK] {month_tag} -> {out_fp} | rows: {len(sample):,}")

    if args.all_months:
        for fp in files:
            make_one(fp, args.n)
        return

    # single month behavior
    if args.month:
        matches = [fp for fp in files if fp.name.startswith(args.month)]
        if not matches:
            raise FileNotFoundError(f"No cleaned file for month {args.month} in {INTERIM_DIR}")
        make_one(matches[0], args.n)
    else:
        make_one(files[0], args.n)

if __name__ == "__main__":
    main()
