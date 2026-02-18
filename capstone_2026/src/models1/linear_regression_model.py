import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from src.models1.feature_engineering import prepare_features
from src.models1.metrics import regression_metrics


def train_linear_regression_model(df, output_path="models/linear_regression_model.pkl"):
    metrics_path = "reports/model_outputs/linear_regression_metrics.csv"
    # Avoid retraining if model already exists
    if os.path.exists(output_path) and os.path.exists(metrics_path):
        print("Linear Regression model already exists. Skipping training.")
        return joblib.load(output_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = regression_metrics(y_test, preds)
    # Save model
    joblib.dump(model, output_path)
    # Save metrics
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print("Linear Regression Model Trained and Saved")
    print(metrics)
    return model
