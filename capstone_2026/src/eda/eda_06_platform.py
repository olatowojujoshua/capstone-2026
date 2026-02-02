from src.eda.eda_utils import save_csv

def run(df):
    platform_summary = (
        df.groupby("hvfhs_license_num", observed=True)
        .base_passenger_fare.agg(["mean", "std", "count"])
        .reset_index()
    )
    save_csv(platform_summary, "platform_fares")
    print("EDA 06 complete")