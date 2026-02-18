import os
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from src.models1.feature_engineering import prepare_features
from src.models1.metrics import regression_metrics


def train_xgboost_model(df, output_path="models/xgboost_model.pkl"):
    metrics_path = "reports/model_outputs/xgboost_model_metrics.csv"
    # Skip training if already exists
    if os.path.exists(output_path) and os.path.exists(metrics_path):
        print("XGBoost model already exists. Skipping training.")
        return joblib.load(output_path)
    print("Preparing features...")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Training XGBoost...")
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = regression_metrics(y_test, preds)
    joblib.dump(model, output_path)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print("XGBoost Model Trained and Saved")
    print(metrics)
    return model
