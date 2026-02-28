import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from src.eda_plots.plot_utils import (
    save_fig, apply_dark_style, add_glow_line, add_gradient_fill,
    FONT_FAMILY,
)

def run():
    df = pd.read_csv("reports/eda/fare_by_month.csv")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = df["month"].values
    y = df["base_passenger_fare"].values
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    fig, ax = plt.subplots(figsize=(11, 6))
    y_pad = (y.max() - y.min()) * 0.2
    y_min = y.min() - y_pad
    y_max = y.max() + y_pad
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0.5, 8.5)
    add_gradient_fill(ax, x_smooth, y_smooth, "#7c4dff", "#00d4ff", alpha=0.22, y_min=y_min)
    add_glow_line(ax, x_smooth, y_smooth, color="#7c4dff", linewidth=2.5)
    ax.scatter(x, y, color="#7c4dff", s=45, zorder=6, edgecolors="white", linewidth=0.6)
    peak_idx = y.argmax()
    ax.annotate(
        f"Peak: ${y[peak_idx]:.1f}",
        xy=(x[peak_idx], y[peak_idx]),
        xytext=(x[peak_idx] - 1.8, y[peak_idx] + 0.8),
        fontsize=10, fontweight="bold",
        color="#ffab40", fontfamily=FONT_FAMILY,
        arrowprops=dict(arrowstyle="->", color="#ffab40", lw=1.5),
        zorder=7,
    )
    valley_idx = y.argmin()
    ax.annotate(
        f"Low: ${y[valley_idx]:.1f}",
        xy=(x[valley_idx], y[valley_idx]),
        xytext=(x[valley_idx] + 1.5, y[valley_idx] - 1.0),
        fontsize=10, fontweight="bold",
        color="#ff4081", fontfamily=FONT_FAMILY,
        arrowprops=dict(arrowstyle="->", color="#ff4081", lw=1.5),
        zorder=7,
    )
    pct_change = (y[-1] - y[0]) / y[0] * 100
    ax.text(
        0.98, 0.05,
        f"Jan->Dec: +{pct_change:.1f}%",
        transform=ax.transAxes,
        fontsize=11, fontweight="bold",
        color="#69f0ae", fontfamily=FONT_FAMILY,
        ha="right", va="bottom",
        bbox=dict(facecolor="#1e1e3a", edgecolor="#69f0ae",
                  alpha=0.7, boxstyle="round,pad=0.4"),
        zorder=8,
    )
    apply_dark_style(
        ax, fig,
        title="Monthly Average Fare Trend (2021)",
        xlabel="Month",
        ylabel="Average Fare ($)",
    )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=10)
    save_fig("monthly_fare_trend")
    print("Monthly trend plot saved")

if __name__ == "__main__":
    run()