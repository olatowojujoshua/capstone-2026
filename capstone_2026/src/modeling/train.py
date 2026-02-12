import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from pathlib import Path

def train_model(X: pd.DataFrame, y: pd.Series, model_dir: Path) -> dict:
    """
    Train a Gradient Boosting regressor and evaluate on a hold-out test set.
    Returns metrics dict and saves model and metrics.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "mean_y_true": float(np.mean(y_test)),
        "mean_y_pred": float(np.mean(y_pred)),
    }

    # Save model and metrics
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "fare_gbt_model.joblib")
    pd.Series(metrics).to_csv(model_dir / "metrics.csv", header=["value"])

    return metrics
