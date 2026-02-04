import pandas as pd
import matplotlib.pyplot as plt
from src.eda_plots.plot_utils import save_fig

def run():
    df = pd.read_csv("reports/eda/hourly_fare_volatility.csv")
    df["hour"] = pd.to_datetime(df["hour"])
    df["std_smooth"] = df["std"].rolling(24).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(df["hour"], df["std_smooth"])
    plt.xlabel("Time")
    plt.ylabel("Fare Std Dev ($)")
    plt.title("Hourly Fare Volatility Over Time")
    save_fig("hourly_fare_volatility_trend")
    print("Volatility trend plot saved")

if __name__ == "__main__":
    run()