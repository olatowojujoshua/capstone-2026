import pandas as pd
import numpy as np
from src.models1.feature_engineering import prepare_features

def evaluate_fairness(model, df):
    df = df.copy()
    X, y = prepare_features(df)
    df["prediction"] = model.predict(X)
    df["error"] = np.abs(df["prediction"] - y)
    # Fairness by Pickup Location
    zone_error = (
        df.groupby("PULocationID")["error"]
        .mean()
        .reset_index()
        .rename(columns={"error": "mean_abs_error"})
    )
    # Fairness by Hour of Day
    df["hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour
    hour_error = (
        df.groupby("hour")["error"]
        .mean()
        .reset_index()
        .rename(columns={"error": "mean_abs_error"})
    )
    print("Fairness evaluation complete")
    return zone_error, hour_error