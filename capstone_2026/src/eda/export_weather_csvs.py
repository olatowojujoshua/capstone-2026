"""
Export Weather CSVs for Dashboard
==================================
Reads the zone_time_features parquet and produces CSV summary files
that the Django dashboard can load without touching the heavy parquet.

Run once (or whenever the parquet is refreshed):
    python capstone_2026/src/eda/export_weather_csvs.py

Output CSVs (all written to  reports/eda/):
    weather_trip_count_rain.csv
    weather_fare_wet.csv
    weather_delay_rain.csv
    weather_trip_count_temp.csv
    weather_volatility_wet.csv
    weather_hourly_demand.csv
    weather_top_zones.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = PROJECT_ROOT / "reports" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "zone_time_features" / "zone_time_features.parquet"


def main():
    print("Loading feature store …")
    df = pd.read_parquet(DATA_PATH)

    # Helper: temperature band
    def _temp_band(row):
        if row["temp_bin_cold"] == 1:
            return "Cold"
        elif row["temp_bin_hot"] == 1:
            return "Hot"
        return "Moderate"

    df["temp_band"] = df.apply(_temp_band, axis=1)
    df["hour"] = pd.to_datetime(df["time_bucket"]).dt.hour

    # 1. Trip Count vs Rain
    grp = df.groupby("is_raining", as_index=False)["trip_count"].mean()
    grp.to_csv(EDA_DIR / "weather_trip_count_rain.csv", index=False)
    print("  ✓ weather_trip_count_rain.csv")

    # 2. Fare per Mile: Dry vs Wet
    grp = df.groupby("is_wet_weather", as_index=False)["med_fare_per_mile"].mean()
    grp.to_csv(EDA_DIR / "weather_fare_wet.csv", index=False)
    print("  ✓ weather_fare_wet.csv")

    # 3. Pickup Delay vs Rain
    grp = df.groupby("is_raining", as_index=False)["avg_pickup_delay_sec"].mean()
    grp.to_csv(EDA_DIR / "weather_delay_rain.csv", index=False)
    print("  ✓ weather_delay_rain.csv")

    # 4. Trip Count by Temperature Band
    grp = df.groupby("temp_band", as_index=False)["trip_count"].mean()
    grp.to_csv(EDA_DIR / "weather_trip_count_temp.csv", index=False)
    print("  ✓ weather_trip_count_temp.csv")

    # 5. Fare Volatility (Std Dev): Dry vs Wet
    grp = df.groupby("is_wet_weather", as_index=False)["med_fare_per_mile"].std()
    grp.columns = ["is_wet_weather", "med_fare_per_mile_std"]
    grp.to_csv(EDA_DIR / "weather_volatility_wet.csv", index=False)
    print("  ✓ weather_volatility_wet.csv")

    # 6. Hourly Demand by Rain Condition
    hourly = df.groupby(["hour", "is_raining"])["trip_count"].mean().unstack().reset_index()
    hourly.columns = ["hour", "no_rain", "rain"]
    hourly = hourly.sort_values("hour")
    hourly.to_csv(EDA_DIR / "weather_hourly_demand.csv", index=False)
    print("  ✓ weather_hourly_demand.csv")

    # 7. Top 10 Zones Most Affected by Wet Weather
    zone_weather = (
        df.groupby(["PULocationID", "is_wet_weather"])["med_fare_per_mile"]
        .mean()
        .unstack()
        .fillna(0)
    )
    zone_weather["wet_minus_dry"] = zone_weather[1] - zone_weather[0]
    top_zones = zone_weather.sort_values("wet_minus_dry", ascending=False).head(10)
    top_zones = top_zones.reset_index()[["PULocationID", "wet_minus_dry"]]

    # Map zone IDs → names
    zones_csv = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "taxi_zone_lookup.csv")
    zone_name_map = dict(zip(zones_csv["LocationID"], zones_csv["Zone"]))
    top_zones["zone_name"] = top_zones["PULocationID"].apply(
        lambda z: zone_name_map.get(int(z), f"Zone {int(z)}")
    )
    top_zones.to_csv(EDA_DIR / "weather_top_zones.csv", index=False)
    print("  ✓ weather_top_zones.csv")

    print(f"\n✅ All weather CSVs saved to: {EDA_DIR}")


if __name__ == "__main__":
    main()
