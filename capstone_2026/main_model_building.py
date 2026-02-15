from src.models1.fare_prediction import train_fare_model
from src.models1.volatility_model import train_volatility_model
from src.models1.fairness_analysis import evaluate_fairness
import pandas as pd
from src.eda.eda_utils import load_all_parquets

def main():
    print("Loading monthly parquet files...")
    df = load_all_parquets()
    print("Training Fare Prediction Model...")
    fare_model = train_fare_model(df)
    print("Preparing hourly volatility dataset...")
    vol_df = pd.read_csv("reports/eda/hourly_fare_volatility.csv")
    print("Training Volatility Model...")
    train_volatility_model(vol_df)
    print("Running Fairness Evaluation...")
    zone_error, hour_error = evaluate_fairness(fare_model, df)
    zone_error.to_csv("reports/model_outputs/zone_error.csv", index=False)
    hour_error.to_csv("reports/model_outputs/hour_error.csv", index=False)
    print("All models built successfully.")

if __name__ == "__main__":
    main()