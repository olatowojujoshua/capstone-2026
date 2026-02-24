import pandas as pd
import matplotlib.pyplot as plt
from src.eda_plots.plot_utils import (
    save_fig, apply_dark_style, ACCENT_COLORS,
    TEXT_COLOR, FONT_FAMILY,
)

def run():
    df = pd.read_csv("reports/eda/fare_by_trip_length.csv")
    order = {"short": 0, "medium": 1, "long": 2}
    df["sort_key"] = df["trip_length_bucket"].map(order)
    df = df.sort_values("sort_key")
    labels = df["trip_length_bucket"].str.title().values
    values = df["mean"].values
    counts = df["count"].values
    colors = [ACCENT_COLORS[0], ACCENT_COLORS[1], ACCENT_COLORS[2]]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(
        labels, values,
        color=colors, width=0.5,
        edgecolor="white", linewidth=0.3,
        zorder=3,
    )
    # Glow shadow
    for bar, color in zip(bars, colors):
        ax.bar(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            width=bar.get_width() * 1.2,
            color=color, alpha=0.08, zorder=2,
        )
    # Value + count labels
    for bar, val, cnt in zip(bars, values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + values.max() * 0.02,
            f"${val:.2f}",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{cnt / 1e6:.1f}M trips",
            ha="center", va="center",
            fontsize=9, color="white", fontfamily=FONT_FAMILY,
            alpha=0.7,
        )
    apply_dark_style(
        ax, fig,
        title="Mean Fare by Trip Length Category",
        xlabel="Trip Length",
        ylabel="Mean Base Fare ($)",
    )
    ax.set_ylim(0, values.max() * 1.2)
    save_fig("fare_by_trip_length")
    print("Trip length plots saved")

if __name__ == "__main__":
    run()