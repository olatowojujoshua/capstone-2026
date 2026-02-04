import pandas as pd
import matplotlib.pyplot as plt
from src.eda_plots.plot_utils import save_fig

def run():
    df = pd.read_csv("reports/eda/pickup_zone_fares.csv")
    df = df.sort_values("mean_fare", ascending=False).head(20)
    plt.figure(figsize=(8, 5))
    plt.barh(df["PULocationID"].astype(str), df["mean_fare"])
    plt.xlabel("Average Fare ($)")
    plt.ylabel("Pickup Zone")
    plt.title("Top 20 Pickup Zones by Average Fare")
    save_fig("top_zones_avg_fare")
    print("Spatial plots saved")

if __name__ == "__main__":
    run()