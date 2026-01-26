import numpy as np
import pandas as pd

# Columns we expect to exist in HVFHS (some may be missing by month/version; we handle safely)
CORE_COLS = [
    "hvfhs_license_num",
    "request_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "sales_tax",
    "bcf",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    "shared_request_flag",
    "wav_request_flag",
    "wav_match_flag",
]

FLAG_COLS = ["shared_request_flag", "wav_request_flag", "wav_match_flag"]

def safe_to_datetime(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

def clean_month_df(df: pd.DataFrame) -> pd.DataFrame:
    # keep only known columns that exist
    keep_cols = [c for c in CORE_COLS if c in df.columns]
    df = df[keep_cols].copy()

    # datetimes
    for c in ["request_datetime", "pickup_datetime", "dropoff_datetime"]:
        safe_to_datetime(df, c)

    # drop rows missing critical fields
    critical = ["pickup_datetime", "PULocationID", "DOLocationID", "trip_miles", "trip_time", "base_passenger_fare"]
    critical = [c for c in critical if c in df.columns]
    df = df.dropna(subset=critical)

    # enforce numeric types
    for c in ["trip_miles", "trip_time", "base_passenger_fare", "tips", "driver_pay",
              "tolls", "sales_tax", "bcf", "congestion_surcharge", "airport_fee"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # remove impossible/invalid trips
    if "trip_miles" in df.columns:
        df = df[df["trip_miles"] > 0]
    if "trip_time" in df.columns:
        df = df[df["trip_time"] > 0]
    if "base_passenger_fare" in df.columns:
        df = df[df["base_passenger_fare"] > 0]

    # pickup delay proxy (seconds)
    if "request_datetime" in df.columns and "pickup_datetime" in df.columns:
        df["pickup_delay_sec"] = (df["pickup_datetime"] - df["request_datetime"]).dt.total_seconds()
        # keep only realistic delays (0 to 2 hours)
        df = df[(df["pickup_delay_sec"].isna()) | ((df["pickup_delay_sec"] >= 0) & (df["pickup_delay_sec"] <= 7200))]

    # fare per mile (useful for outlier control + vehicle tier proxy)
    df["fare_per_mile"] = df["base_passenger_fare"] / df["trip_miles"]

    # cap extreme outliers (winsorize) — conservative caps
    for c in ["trip_miles", "trip_time", "base_passenger_fare", "fare_per_mile"]:
        if c in df.columns:
            lo = df[c].quantile(0.005)
            hi = df[c].quantile(0.995)
            df[c] = df[c].clip(lower=lo, upper=hi)

    # flags: normalize to 0/1
    for c in FLAG_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().map({"Y": 1, "N": 0}).fillna(0).astype("int8")

    # categorical compression
    if "hvfhs_license_num" in df.columns:
        df["hvfhs_license_num"] = df["hvfhs_license_num"].astype("category")
    for c in ["PULocationID", "DOLocationID"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")

    return df
