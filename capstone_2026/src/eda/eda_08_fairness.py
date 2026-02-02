import numpy as np
from src.eda.eda_utils import save_csv

def gini(x):
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n+1) * x))) / (n * np.sum(x)) - (n + 1) / n

def run(df):
    zone_fares = df.groupby("PULocationID")["base_passenger_fare"].mean().reset_index()
    gini_index = gini(zone_fares["base_passenger_fare"].values)
    save_csv(zone_fares, "zone_average_fares")
    print("EDA 08 complete - Gini:", round(gini_index, 4))