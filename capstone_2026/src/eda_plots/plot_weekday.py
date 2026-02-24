import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.eda_plots.plot_utils import (
    save_fig, apply_dark_style, ACCENT_COLORS,
    TEXT_COLOR, FONT_FAMILY, GRID_COLOR,
)

def run():
    df = pd.read_csv("reports/eda/fare_by_weekday_sampled.csv")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    day_abbr = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fare_map = dict(zip(df["weekday"], df["base_passenger_fare"]))
    values = [fare_map.get(d, 0) for d in day_order]
    colors = [ACCENT_COLORS[0]] * 5 + [ACCENT_COLORS[2]] * 2
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        day_abbr[::-1], values[::-1],
        color=colors[::-1], height=0.55,
        edgecolor="white", linewidth=0.3,
        zorder=3,
    )
    # Glow shadow
    for bar, color in zip(bars, colors[::-1]):
        ax.barh(
            bar.get_y() + bar.get_height() / 2,
            bar.get_width(),
            height=bar.get_height() * 1.2,
            color=color, alpha=0.08, zorder=2,
        )
    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"${width:.2f}",
            va="center", ha="left",
            fontsize=11, fontweight="bold",
            color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        )
    # Highlight highest fare day
    max_idx = int(np.argmax(values[::-1]))
    bars[max_idx].set_edgecolor("#ffab40")
    bars[max_idx].set_linewidth(1.5)
    apply_dark_style(
        ax, fig,
        title="Average Fare by Day of Week",
        xlabel="Average Fare ($)",
        ylabel="",
    )
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.4, alpha=0.5)
    ax.grid(False, axis="y")
    # Add weekend/weekday legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT_COLORS[0], label="Weekday"),
        Patch(facecolor=ACCENT_COLORS[2], label="Weekend"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=10, framealpha=0.6,
        facecolor="#13132b", edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        bbox_to_anchor=(1.0, 1.12),
        ncol=2,
    )
    x_max = max(values) * 1.1
    ax.set_xlim(0, x_max)
    save_fig("avg_fare_by_weekday")
    print("Weekday fare plot saved")

if __name__ == "__main__":
    run()