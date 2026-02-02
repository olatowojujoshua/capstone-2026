import pandas as pd
from src.eda.eda_utils import save_csv

def run(df):
    df["fare_per_mile"] = df["base_passenger_fare"] / df["trip_miles"]
    df["trip_length_bucket"] = pd.qcut(
        df["trip_miles"],
        q=[0, .33, .66, 1.0],
        labels=["short", "medium", "long"]
    )
    summary = (
        df.groupby("trip_length_bucket", observed=True)
        .base_passenger_fare.agg(["mean", "std", "count"])
        .reset_index()
    )
    save_csv(summary, "fare_by_trip_length")
    print("EDA 04 complete")