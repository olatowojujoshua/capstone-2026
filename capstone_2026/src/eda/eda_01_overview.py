import pandas as pd
from src.eda.eda_utils import save_csv

FARE_KEYWORDS = (
    "fare",
    "amount",
    "distance",
    "mile",
    "time",
    "duration",
    "pay"
)

def run(df):
    overview = pd.DataFrame([{
        "rows": len(df),
        "columns": df.shape[1],
        "start_date": df["pickup_datetime"].min(),
        "end_date": df["pickup_datetime"].max()
    }])
    save_csv(overview, "dataset_overview")
    missing = (
        df.isna()
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_pct"})
    )
    save_csv(missing, "missing_values")

    # Step 1: sample ROWS
    sample_rows = df.sample(frac=0.01, random_state=42)

    # Step 2: detect numeric columns on sample
    numeric_cols = [
        c for c in sample_rows.columns
        if sample_rows[c].dtype.kind in "if"
        and any(k in c.lower() for k in FARE_KEYWORDS)
    ]
    if not numeric_cols:
        print("No numeric fare-related columns detected")
        return

    # Step 3: aggregate on sampled numeric data
    numeric_summary = (
        sample_rows[numeric_cols]
        .dropna()
        .agg(["mean", "std", "min", "median", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    save_csv(numeric_summary, "numeric_summary_sampled")
    print("EDA 01 complete")