import pandas as pd
import matplotlib.pyplot as plt
from src.eda_plots.plot_utils import save_fig

def run():
    df = pd.read_csv("reports/eda/fare_components.csv")
    plt.figure()
    plt.bar(df["component"], df["average_amount"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Average Amount ($)")
    plt.title("Average Fare Components")
    save_fig("fare_components")
    print("Fare breakdown plot saved")

if __name__ == "__main__":
    run()