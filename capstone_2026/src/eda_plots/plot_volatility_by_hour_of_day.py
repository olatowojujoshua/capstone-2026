import pandas as pd
import matplotlib.pyplot as plt
from src.eda_plots.plot_utils import save_fig

def run():
    df = pd.read_csv("reports/eda/hourly_fare_volatility.csv")
    df["hour"] = pd.to_datetime(df["hour"])
    df["hour_of_day"] = df["hour"].dt.hour
    hourly = (
        df.groupby("hour_of_day")["std"]
        .mean()
        .reset_index()
    )
    plt.figure()
    plt.plot(hourly["hour_of_day"], hourly["std"])
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Fare Std Dev ($)")
    plt.title("Average Fare Volatility by Hour of Day")
    save_fig("avg_volatility_by_hour_of_day")
    print("Hourly-of-day volatility plot saved")

if __name__ == "__main__":
    run()