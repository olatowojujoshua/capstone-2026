import pandas as pd
import matplotlib.pyplot as plt
from src.eda_plots.plot_utils import save_fig

def run():
    df = pd.read_csv("reports/eda/fare_by_trip_length.csv")
    plt.figure()
    plt.bar(df["trip_length_bucket"], df["mean"])
    plt.xlabel("Trip Length")
    plt.ylabel("Mean Base Fare ($)")
    plt.title("Fare by Trip Length Category")
    save_fig("fare_by_trip_length")
    print("Trip length plots saved")

if __name__ == "__main__":
    run()