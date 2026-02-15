import pandas as pd

def create_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    return df

def create_trip_features(df):
    df["fare_per_mile"] = df["base_passenger_fare"] / df["trip_miles"]
    df["is_short_trip"] = (df["trip_miles"] < 2).astype(int)
    return df

def prepare_features(df):
    df = create_time_features(df)
    df = create_trip_features(df)
    features = [
        "trip_miles",
        "hour",
        "day_of_week",
        "month"
    ]
    X = df[features]
    y = df["base_passenger_fare"]
    return X, y