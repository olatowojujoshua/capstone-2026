import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    abs_errors = np.abs(y_true - y_pred)
    p90_abs_err = np.percentile(abs_errors, 90)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "p90_abs_err": p90_abs_err
    }
    return metrics