import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.eda_plots.plot_utils import (
    save_fig, apply_dark_style, CYAN_PURPLE_CMAP, NEON_CMAP,
    TEXT_COLOR, FONT_FAMILY, GRID_COLOR,
    BG_COLOR
)

def run():
    df = pd.read_csv("reports/eda/dropoff_zone_fares.csv")
    df = df.sort_values("mean_fare", ascending=False)
    top15 = df.head(15).sort_values("mean_fare", ascending=True)
    bottom15 = df.tail(15).sort_values("mean_fare", ascending=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    fig.patch.set_facecolor(BG_COLOR)
    norm_top = mcolors.Normalize(vmin=top15["mean_fare"].min(), vmax=top15["mean_fare"].max())
    colors_top = [NEON_CMAP(norm_top(v)) for v in top15["mean_fare"]]
    bars_top = ax1.barh(
        top15["DOLocationID"].astype(str).apply(lambda z: f"Zone {z}"),
        top15["mean_fare"],
        color=colors_top, height=0.65,
        edgecolor="white", linewidth=0.3,
        zorder=3,
    )
    for bar, color in zip(bars_top, colors_top):
        ax1.barh(
            bar.get_y() + bar.get_height() / 2,
            bar.get_width(),
            height=bar.get_height() * 1.2,
            color=color, alpha=0.08, zorder=2,
        )
    for bar in bars_top:
        width = bar.get_width()
        ax1.text(
            width + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"${width:.1f}",
            va="center", ha="left",
            fontsize=9, fontweight="bold",
            color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        )
    bars_top[-1].set_edgecolor("#ffab40")
    bars_top[-1].set_linewidth(1.5)
    apply_dark_style(
        ax1, fig,
        title="Top 15 Highest-Fare Dropoff Zones",
        xlabel="Average Fare ($)",
        ylabel="",
        title_size=14,
    )
    ax1.grid(True, axis="x", color=GRID_COLOR, linewidth=0.4, alpha=0.5)
    ax1.grid(False, axis="y")
    ax1.set_xlim(0, top15["mean_fare"].max() * 1.15)
    norm_bot = mcolors.Normalize(vmin=bottom15["mean_fare"].min(), vmax=bottom15["mean_fare"].max())
    colors_bot = [CYAN_PURPLE_CMAP(norm_bot(v)) for v in bottom15["mean_fare"]]
    bars_bot = ax2.barh(
        bottom15["DOLocationID"].astype(str).apply(lambda z: f"Zone {z}"),
        bottom15["mean_fare"],
        color=colors_bot, height=0.65,
        edgecolor="white", linewidth=0.3,
        zorder=3,
    )
    for bar, color in zip(bars_bot, colors_bot):
        ax2.barh(
            bar.get_y() + bar.get_height() / 2,
            bar.get_width(),
            height=bar.get_height() * 1.2,
            color=color, alpha=0.08, zorder=2,
        )
    for bar in bars_bot:
        width = bar.get_width()
        ax2.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"${width:.1f}",
            va="center", ha="left",
            fontsize=9, fontweight="bold",
            color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        )
    apply_dark_style(
        ax2, fig,
        title="Top 15 Lowest-Fare Dropoff Zones",
        xlabel="Average Fare ($)",
        ylabel="",
        title_size=14,
    )
    ax2.grid(True, axis="x", color=GRID_COLOR, linewidth=0.4, alpha=0.5)
    ax2.grid(False, axis="y")
    ax2.set_xlim(0, bottom15["mean_fare"].max() * 1.25)
    fig.suptitle(
        "Dropoff Zone Fare Extremes",
        fontsize=19, fontweight="bold",
        color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        y=1.02,
    )
    save_fig("dropoff_zones_fare")
    print("Dropoff zones fare plot saved")

if __name__ == "__main__":
    run()