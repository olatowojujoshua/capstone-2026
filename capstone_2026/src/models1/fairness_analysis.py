import pandas as pd
import numpy as np
from src.models1.feature_engineering import prepare_features
import os

def evaluate_fairness(model, df):
    zone_error_path = "reports/model_outputs/zone_error.csv"
    hour_error_path = "reports/model_outputs/hour_error.csv"
    if os.path.exists(zone_error_path) and os.path.exists(hour_error_path):
        print("Fareness evaluation already exists...")
        return
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
    zone_error.to_csv("reports/model_outputs/zone_error.csv", index=False)
    hour_error.to_csv("reports/model_outputs/hour_error.csv", index=False)
    print("Fairness evaluation complete")