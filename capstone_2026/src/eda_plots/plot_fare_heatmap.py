import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from src.eda_plots.plot_utils import (
    save_fig, apply_dark_style,
    SUBTEXT_COLOR, FONT_FAMILY
)

def run():
    df = pd.read_csv("reports/eda/hourly_fare_volatility.csv")
    df["hour"] = pd.to_datetime(df["hour"])
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    heatmap_data = df.pivot_table(
        values="mean", index="hour_of_day", columns="day_of_week", aggfunc="mean"
    )
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hour_labels = []
    for h in range(24):
        if h == 0:
            hour_labels.append("12 AM")
        elif h < 12:
            hour_labels.append(f"{h} AM")
        elif h == 12:
            hour_labels.append("12 PM")
        else:
            hour_labels.append(f"{h - 12} PM")
    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fare_heat", ["#0a0a2e", "#00d4ff", "#7c4dff", "#ff4081"]
    )
    im = ax.imshow(
        heatmap_data.values, aspect="auto", cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_labels, fontsize=11, fontfamily=FONT_FAMILY, color=SUBTEXT_COLOR)
    ax.set_yticks(range(24))
    ax.set_yticklabels(hour_labels, fontsize=9, fontfamily=FONT_FAMILY, color=SUBTEXT_COLOR)
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data.values[i, j]
            norm_val = (val - heatmap_data.values.min()) / (
                heatmap_data.values.max() - heatmap_data.values.min()
            )
            text_color = "#0a0a14" if norm_val > 0.65 else "white"
            ax.text(
                j, i, f"${val:.1f}",
                ha="center", va="center",
                fontsize=8, fontweight="600",
                color=text_color, fontfamily=FONT_FAMILY,
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Average Fare ($)", fontsize=11, color=SUBTEXT_COLOR, fontfamily=FONT_FAMILY)
    cbar.ax.tick_params(colors=SUBTEXT_COLOR, labelsize=9)
    apply_dark_style(
        ax, fig,
        title="Fare Heatmap - Hour of Day x Day of Week",
        xlabel="",
        ylabel="",
    )
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 24, 1), minor=True)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_fig("fare_heatmap_hour_weekday")
    print("Fare heatmap plot saved")

if __name__ == "__main__":
    run()