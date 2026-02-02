import pandas as pd
from src.eda.eda_utils import save_csv

def run(df):
    pickup_ts = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    # Hourly aggregation
    hourly = (
        df
        .groupby(pickup_ts.dt.hour)
        ["base_passenger_fare"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"pickup_datetime": "hour"})
    )

    # Weekday aggregation
    sample = df.sample(frac=0.02, random_state=42)
    weekday = (
        sample
        .groupby(pd.to_datetime(sample["pickup_datetime"]).dt.day_name())
        ["base_passenger_fare"]
        .mean()
        .reset_index()
        .rename(columns={"pickup_datetime": "weekday"})
    )

    # Monthly aggregation
    monthly = (
        df
        .groupby(pickup_ts.dt.month)
        ["base_passenger_fare"]
        .mean()
        .reset_index()
        .rename(columns={"pickup_datetime": "month"})
    )
    save_csv(hourly, "fare_by_hour")
    save_csv(weekday, "fare_by_weekday_sampled")
    save_csv(monthly, "fare_by_month")
    print("EDA 02 complete")