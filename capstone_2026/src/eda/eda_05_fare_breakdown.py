from src.eda.eda_utils import save_csv

FARE_COLS = [
    "base_passenger_fare",
    "congestion_surcharge",
    "airport_fee",
    "tolls",
    "sales_tax",
    "tips"
]

def run(df):
    breakdown = df[FARE_COLS].mean().reset_index()
    breakdown.columns = ["component", "average_amount"]
    save_csv(breakdown, "fare_components")
    print("EDA 05 complete")