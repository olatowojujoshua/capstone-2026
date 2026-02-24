import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from src.eda_plots.plot_utils import (
    save_fig, apply_dark_style, add_glow_line, add_gradient_fill,
    FONT_FAMILY,
)

def run():
    df = pd.read_csv("reports/eda/fare_by_hour.csv")
    x = df["hour"].values
    y = df["mean"].values
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    fig, ax = plt.subplots(figsize=(11, 6))
    y_pad = (y.max() - y.min()) * 0.15
    y_min = y.min() - y_pad
    y_max = y.max() + y_pad
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, 23.5)
    add_gradient_fill(ax, x_smooth, y_smooth, "#00d4ff", "#7c4dff", alpha=0.25, y_min=y_min)
    add_glow_line(ax, x_smooth, y_smooth, color="#00d4ff", linewidth=2.5)
    ax.scatter(x, y, color="#00d4ff", s=30, zorder=6, edgecolors="white", linewidth=0.5)
    peak_idx = y.argmax()
    ax.annotate(
        f"Peak: ${y[peak_idx]:.1f}",
        xy=(x[peak_idx], y[peak_idx]),
        xytext=(x[peak_idx] + 2.5, y[peak_idx] + 0.5),
        fontsize=10, fontweight="bold",
        color="#ffab40", fontfamily=FONT_FAMILY,
        arrowprops=dict(arrowstyle="->", color="#ffab40", lw=1.5),
        zorder=7,
    )
    valley_idx = y.argmin()
    ax.annotate(
        f"Low: ${y[valley_idx]:.1f}",
        xy=(x[valley_idx], y[valley_idx]),
        xytext=(x[valley_idx] + 2.5, y[valley_idx] - 0.8),
        fontsize=10, fontweight="bold",
        color="#ff4081", fontfamily=FONT_FAMILY,
        arrowprops=dict(arrowstyle="->", color="#ff4081", lw=1.5),
        zorder=7,
    )
    apply_dark_style(
        ax, fig,
        title="Average Fare by Hour of Day",
        xlabel="Hour of Day",
        ylabel="Average Fare ($)",
    )
    # AM/PM hour labels
    hour_labels = []
    for h in range(0, 24, 3):
        if h == 0:
            hour_labels.append("12 AM")
        elif h < 12:
            hour_labels.append(f"{h} AM")
        elif h == 12:
            hour_labels.append("12 PM")
        else:
            hour_labels.append(f"{h - 12} PM")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels(hour_labels, fontsize=9)
    save_fig("avg_fare_by_hour")
    print("Temporal plots saved")

if __name__ == "__main__":
    run()