import pandas as pd
import joblib, os, json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from src.models1.metrics import regression_metrics

def train_volatility_model(df, output_path="models/volatility_model.pkl"):
    metrics_path = "reports/model_outputs/volatility_model_metrics.csv"
    # If model and metrics already exist, load and return
    if os.path.exists(output_path) and os.path.exists(metrics_path):
        print("Volatility model already exists. Loading existing model...")
        model = joblib.load(output_path)
        return model
    # ---- Feature Engineering ----
    df = df.copy()  # avoid modifying original df
    df["hour"] = pd.to_datetime(df["hour"])
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["month"] = df["hour"].dt.month
    features = ["hour_of_day", "day_of_week", "month"]
    X = df[features]
    y = df["std"]
    # ---- Train/Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # ---- Model Training ----
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = regression_metrics(y_test, preds)
    # ---- Ensure directories exist ----
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/model_outputs", exist_ok=True)
    # ---- Save Model + Metrics ----
    joblib.dump(model, output_path)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    print(json.dumps(metrics, indent=4))
    print("Volatility Model Trained & Saved")
    print("------------************------------")
    return model