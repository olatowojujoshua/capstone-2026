import pandas as pd
from src.eda.eda_utils import save_csv

def run(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.floor("h")
    volatility = (
        df.groupby("hour")
        .base_passenger_fare.agg(["mean", "std"])
        .reset_index()
    )
    volatility["cv"] = volatility["std"] / volatility["mean"]
    save_csv(volatility, "hourly_fare_volatility")
    print("EDA 07 complete")