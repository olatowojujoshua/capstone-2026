import pandas as pd
import requests
from src.config import PROCESSED_DIR


def fetch_weather_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "rain",
            "snowfall",
            "relative_humidity_2m",
            "wind_speed_10m",
        ],
        "timezone": "America/New_York",
    }

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def build_weather_2021() -> pd.DataFrame:
    month_ranges = [
        ("2021-01-01", "2021-01-31"),
        ("2021-02-01", "2021-02-28"),
        ("2021-03-01", "2021-03-31"),
        ("2021-04-01", "2021-04-30"),
        ("2021-05-01", "2021-05-31"),
        ("2021-06-01", "2021-06-30"),
        ("2021-07-01", "2021-07-31"),
        ("2021-08-01", "2021-08-31"),
        ("2021-09-01", "2021-09-30"),
        ("2021-10-01", "2021-10-31"),
        ("2021-11-01", "2021-11-30"),
        ("2021-12-01", "2021-12-31"),
    ]

    parts = []
    for start_date, end_date in month_ranges:
        print(f"Fetching {start_date} to {end_date}...")
        chunk = fetch_weather_chunk(start_date, end_date)
        print(f"  Rows fetched: {len(chunk)}")
        parts.append(chunk)

    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")

    df["pickup_hour"] = df["time"].dt.floor("h")
    df["is_raining"] = (df["rain"].fillna(0) > 0).astype(int)
    df["is_snowing"] = (df["snowfall"].fillna(0) > 0).astype(int)
    df["is_wet_weather"] = (df["precipitation"].fillna(0) > 0).astype(int)
    df["temp_bin_cold"] = (df["temperature_2m"] < 5).astype(int)
    df["temp_bin_hot"] = (df["temperature_2m"] > 25).astype(int)

    keep_cols = [
        "time",
        "pickup_hour",
        "temperature_2m",
        "precipitation",
        "rain",
        "snowfall",
        "relative_humidity_2m",
        "wind_speed_10m",
        "is_raining",
        "is_snowing",
        "is_wet_weather",
        "temp_bin_cold",
        "temp_bin_hot",
    ]
    return df[keep_cols]


def main():
    print("Building full-year weather table...")
    df = build_weather_2021()

    output_path = PROCESSED_DIR / "weather_hourly_2021.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved file to: {output_path}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()