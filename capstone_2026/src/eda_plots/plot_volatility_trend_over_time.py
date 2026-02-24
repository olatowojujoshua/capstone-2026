import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from src.eda_plots.plot_utils import (
    save_fig, apply_dark_style, add_glow_line, add_gradient_fill,
    SUBTEXT_COLOR, FONT_FAMILY, GRID_COLOR,
)

def run():
    df = pd.read_csv("reports/eda/hourly_fare_volatility.csv")
    df["hour"] = pd.to_datetime(df["hour"])
    df["std_smooth"] = df["std"].rolling(24, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x_num = mdates.date2num(df["hour"].values)
    y = df["std_smooth"].values
    add_gradient_fill(ax, x_num, y, "#7c4dff", "#ff4081", alpha=0.18)
    ax.plot(df["hour"], df["std"], color="#7c4dff", alpha=0.1, linewidth=0.5, zorder=2)
    add_glow_line(ax, df["hour"].values, y, color="#7c4dff", linewidth=2.0)
    y_start = y[~np.isnan(y)][0]
    y_end = y[~np.isnan(y)][-1]
    if y_end > y_start:
        trend_text = f"↑ Volatility rising ({y_start:.1f} → {y_end:.1f})"
        trend_color = "#ff4081"
    else:
        trend_text = f"↓ Volatility falling ({y_start:.1f} → {y_end:.1f})"
        trend_color = "#69f0ae"
    ax.text(
        0.98, 0.95, trend_text,
        transform=ax.transAxes,
        fontsize=11, fontweight="bold",
        color=trend_color, fontfamily=FONT_FAMILY,
        ha="right", va="top",
        bbox=dict(facecolor=GRID_COLOR, edgecolor=trend_color, alpha=0.7, boxstyle="round,pad=0.4"),
        zorder=8,
    )
    apply_dark_style(
        ax, fig,
        title="Hourly Fare Volatility Trend Over Time",
        xlabel="",
        ylabel="Fare Std Dev ($)",
    )
    # Month-based x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=0)
    for label in ax.get_xticklabels():
        label.set_fontfamily(FONT_FAMILY)
        label.set_fontsize(10)
        label.set_color(SUBTEXT_COLOR)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)
    save_fig("hourly_fare_volatility_trend")
    print("Volatility trend plot saved")

if __name__ == "__main__":
    run()