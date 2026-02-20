from src.eda import (
    eda_01_overview,
    eda_02_temporal,
    eda_03_spatial,
    eda_04_trip_features,
    eda_05_fare_breakdown,
    eda_06_platform,
    eda_07_volatility,
    eda_08_fairness
)
from src.eda.eda_utils import load_all_parquets

def main():
    df = load_all_parquets()
    eda_01_overview.run(df)
    eda_02_temporal.run(df)
    eda_03_spatial.run(df)
    eda_04_trip_features.run(df)
    eda_05_fare_breakdown.run(df)
    eda_06_platform.run(df)
    eda_07_volatility.run(df)
    eda_08_fairness.run(df)
    print("All EDA completed successfully")

if __name__ == "__main__":
    main()