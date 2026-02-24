import pandas as pd
import matplotlib.pyplot as plt
from src.eda_plots.plot_utils import (
    save_fig, apply_dark_style, ACCENT_COLORS, TEXT_COLOR,
    FONT_FAMILY
)

def run():
    df = pd.read_csv("reports/eda/fare_components.csv")
    df["label"] = (
        df["component"]
        .str.replace("_", " ", regex=False)
        .str.title()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        df["label"],
        df["average_amount"],
        color=ACCENT_COLORS[: len(df)],
        width=0.55,
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )

    # Glow shadow behind each bar
    for bar, color in zip(bars, ACCENT_COLORS):
        ax.bar(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            width=bar.get_width() * 1.15,
            color=color,
            alpha=0.08,
            zorder=2,
        )

    # Value labels above bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(df["average_amount"]) * 0.02,
            f"${height:.2f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        )

    apply_dark_style(
        ax, fig,
        title="Average Fare Components",
        xlabel="Component",
        ylabel="Average Amount ($)",
    )
    ax.set_ylim(0, max(df["average_amount"]) * 1.15)
    ax.set_xticklabels(df["label"], rotation=25, ha="right", fontsize=10)
    save_fig("fare_components")
    print("Fare breakdown plot saved")

if __name__ == "__main__":
    run()