from src.eda.eda_utils import save_csv

FARE_CANDIDATES = [
    "base_passenger_fare",
    "total_fare",
    "base_passenger_fare"
]

def detect_fare_column(df):
    for col in FARE_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError("No fare column found in dataset")

def run(df):
    fare_col = detect_fare_column(df)

    # Pickup zone aggregation
    pickup_zone = (
        df
        .groupby("PULocationID")[fare_col]
        .agg(mean_fare="mean", std_fare="std", trip_count="count")
        .reset_index()
    )

    # Dropoff zone aggregation
    dropoff_zone = (
        df
        .groupby("DOLocationID")[fare_col]
        .agg(mean_fare="mean", std_fare="std", trip_count="count")
        .reset_index()
    )
    save_csv(pickup_zone, "pickup_zone_fares")
    save_csv(dropoff_zone, "dropoff_zone_fares")
    print("EDA 03 complete")