import pandas as pd
import joblib, os, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.models1.feature_engineering import prepare_features
from src.models1.metrics import regression_metrics

def train_fare_model(df, output_path="models/fare_model.pkl"):
    metrics_path = "reports/model_outputs/fare_model_metrics.csv"
    # If model and metrics already exist, load and return
    if os.path.exists(output_path) and os.path.exists(metrics_path):
        print("Fare model already exists. Loading existing model...")
        model = joblib.load(output_path)
        return model
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = regression_metrics(y_test, preds)
    joblib.dump(model, output_path)
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    print(json.dumps(metrics, indent=4))
    print("Fare Model Saved")
    print("------------************------------")
    return model