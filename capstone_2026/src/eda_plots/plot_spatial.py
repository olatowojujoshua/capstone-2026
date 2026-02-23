import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from src.eda_plots.plot_utils import (
    save_fig, apply_dark_style, CYAN_PURPLE_CMAP,
    TEXT_COLOR, SUBTEXT_COLOR, FONT_FAMILY, GRID_COLOR,
)


def run():
    df = pd.read_csv("reports/eda/pickup_zone_fares.csv")
    df = df.sort_values("mean_fare", ascending=False).head(15)
    df = df.sort_values("mean_fare", ascending=True)  # bottom-up for barh

    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize fares for colormap
    norm = mcolors.Normalize(vmin=df["mean_fare"].min(), vmax=df["mean_fare"].max())
    colors = [CYAN_PURPLE_CMAP(norm(v)) for v in df["mean_fare"]]

    bars = ax.barh(
        df["PULocationID"].astype(str),
        df["mean_fare"],
        color=colors,
        height=0.65,
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )

    # Glow behind bars
    for bar, color in zip(bars, colors):
        ax.barh(
            bar.get_y() + bar.get_height() / 2,
            bar.get_width(),
            height=bar.get_height() * 1.2,
            color=color,
            alpha=0.08,
            zorder=2,
        )

    # Value labels at end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"${width:.1f}",
            va="center", ha="left",
            fontsize=9, fontweight="bold",
            color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        )

    # Highlight #1 zone
    top_bar = bars[-1]
    top_bar.set_edgecolor("#ffab40")
    top_bar.set_linewidth(1.5)

    apply_dark_style(
        ax, fig,
        title="Top 15 Pickup Zones by Average Fare",
        xlabel="Average Fare ($)",
        ylabel="Pickup Zone ID",
    )
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.4, alpha=0.5)
    ax.grid(False, axis="y")
    ax.set_xlim(0, df["mean_fare"].max() * 1.15)

    save_fig("top_zones_avg_fare")
    print("✓ Spatial plots saved")


if __name__ == "__main__":
    run()