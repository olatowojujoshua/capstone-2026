from src.models1.fare_prediction import train_fare_model
from src.models1.volatility_model import train_volatility_model
from src.models1.fairness_analysis import evaluate_fairness
from src.models1.linear_regression_model import train_linear_regression_model
from src.models1.xgboost_model import train_xgboost_model
import pandas as pd
from src.eda.eda_utils import load_all_parquets

def main():
    print("Loading monthly parquet files...")
    df = load_all_parquets()
    print("Training Fare Prediction Model...")
    fare_model = train_fare_model(df)
    print("Training Linear Regression Model...")
    train_linear_regression_model(df)
    print("Training XGBoost Model...")
    train_xgboost_model(df)
    vol_df = pd.read_csv("reports/eda/hourly_fare_volatility.csv")
    print("Training Volatility Model...")
    train_volatility_model(vol_df)
    print("Running Fairness Evaluation...")
    evaluate_fairness(fare_model, df)
    print("All models built successfully.")

if __name__ == "__main__":
    main()