import pandas as pd
import matplotlib.pyplot as plt
from src.eda_plots.plot_utils import (
    save_fig, apply_dark_style, ACCENT_COLORS,
    TEXT_COLOR, FONT_FAMILY,
    BG_COLOR
)

PLATFORM_MAP = {
    "HV0002": "Juno",
    "HV0003": "Uber",
    "HV0004": "Via",
    "HV0005": "Lyft",
}
PLATFORM_COLORS = {
    "Uber":  "#00d4ff",
    "Lyft":  "#ff4081",
    "Via":   "#ffab40",
    "Juno":  "#7c4dff",
}

def run():
    df = pd.read_csv("reports/eda/platform_fares.csv")
    df["platform"] = df["hvfhs_license_num"].map(PLATFORM_MAP)
    df = df.sort_values("mean", ascending=True)
    platforms = df["platform"].values
    fares = df["mean"].values
    colors = [PLATFORM_COLORS.get(p, ACCENT_COLORS[0]) for p in platforms]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG_COLOR)
    # ── Left panel: Average Fare ──
    bars1 = ax1.bar(
        platforms, fares,
        color=colors, width=0.5,
        edgecolor="white", linewidth=0.3,
        zorder=3,
    )
    # Glow shadow
    for bar, color in zip(bars1, colors):
        ax1.bar(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            width=bar.get_width() * 1.2,
            color=color, alpha=0.08, zorder=2,
        )
    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + fares.max() * 0.02,
            f"${height:.2f}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        )
    apply_dark_style(
        ax1, fig,
        title="Average Fare by Platform",
        xlabel="Platform",
        ylabel="Average Fare ($)",
        title_size=15,
    )
    ax1.set_ylim(0, fares.max() * 1.18)

    # ── Right panel: Trip Count ──
    df_count = df.sort_values("count", ascending=True)
    platforms_c = df_count["platform"].values
    counts_c = df_count["count"].values
    colors_c = [PLATFORM_COLORS.get(p, ACCENT_COLORS[0]) for p in platforms_c]
    bars2 = ax2.bar(
        platforms_c, counts_c,
        color=colors_c, width=0.5,
        edgecolor="white", linewidth=0.3,
        zorder=3,
    )
    # Glow shadow
    for bar, color in zip(bars2, colors_c):
        ax2.bar(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            width=bar.get_width() * 1.2,
            color=color, alpha=0.08, zorder=2,
        )
    # Value labels (in millions)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + counts_c.max() * 0.02,
            f"{height / 1e6:.1f}M",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        )
    apply_dark_style(
        ax2, fig,
        title="Trip Count by Platform",
        xlabel="Platform",
        ylabel="Trip Count",
        title_size=15,
    )
    ax2.set_ylim(0, counts_c.max() * 1.18)
    # Format y-axis in millions
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M")
    )
    fig.suptitle(
        "Platform Comparison",
        fontsize=19, fontweight="bold",
        color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        y=1.02,
    )
    save_fig("platform_comparison")
    print("Platform comparison plot saved")

if __name__ == "__main__":
    run()