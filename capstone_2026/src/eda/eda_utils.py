import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/interim")
REPORT_PATH = Path("reports/eda")
REPORT_PATH.mkdir(parents=True, exist_ok=True)

def load_all_parquets():
    files = sorted(DATA_PATH.glob("*_clean.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df

def save_csv(df, name):
    df.to_csv(REPORT_PATH / f"{name}.csv", index=False)