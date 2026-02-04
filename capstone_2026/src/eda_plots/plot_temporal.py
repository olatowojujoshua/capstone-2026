import pandas as pd
import matplotlib.pyplot as plt
from src.eda_plots.plot_utils import save_fig

def run():
    df = pd.read_csv("reports/eda/fare_by_hour.csv")
    plt.figure()
    plt.plot(df["hour"], df["mean"])
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Fare ($)")
    plt.title("Average Fare by Hour of Day")
    save_fig("avg_fare_by_hour")
    print("Temporal plots saved")

if __name__ == "__main__":
    run()