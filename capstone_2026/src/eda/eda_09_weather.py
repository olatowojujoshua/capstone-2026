"""
EDA 09 – Weather Impact Visualizations
=======================================
Produces seven high-fidelity dark-theme charts that show how weather
conditions affect ride-hailing demand, pricing, delays, and volatility.

All figures are saved to  reports/figures/  and follow the same visual
language as the rest of the project (plot_utils dark-neon palette).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from scipy.interpolate import make_interp_spline

# ── project-wide plot helpers ────────────────────────────────────────
import sys, os
# Add capstone_2026 root to sys.path so `src.*` imports work when the
# script is executed directly (python src/eda/eda_09_weather.py).
_CAPSTONE_ROOT = Path(__file__).resolve().parents[2]
if str(_CAPSTONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAPSTONE_ROOT))

from src.eda_plots.plot_utils import (
    save_fig, apply_dark_style, add_glow_line, add_gradient_fill,
    ACCENT_COLORS, TEXT_COLOR, SUBTEXT_COLOR, FONT_FAMILY,
    BG_COLOR, PANEL_COLOR, GRID_COLOR, CYAN_PURPLE_CMAP, NEON_CMAP,
)

# ── paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
EDA_DIR      = PROJECT_ROOT / "reports" / "eda"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "zone_time_features" / "zone_time_features.parquet"

# ── colour shortcuts ─────────────────────────────────────────────────
C_CYAN   = ACCENT_COLORS[0]   # #00d4ff
C_PURPLE = ACCENT_COLORS[1]   # #7c4dff
C_PINK   = ACCENT_COLORS[2]   # #ff4081
C_AMBER  = ACCENT_COLORS[3]   # #ffab40
C_GREEN  = ACCENT_COLORS[4]   # #69f0ae
C_BLUE   = ACCENT_COLORS[5]   # #40c4ff

WEATHER_PAIR = [C_CYAN, C_PINK]        # dry / wet  or  no-rain / rain
TEMP_TRIO    = [C_BLUE, C_GREEN, C_PINK]  # cold / moderate / hot


# ═══════════════════════════════════════════════════════════════════════
#  Helper: vertical bar chart with glow + value labels
# ═══════════════════════════════════════════════════════════════════════
def _styled_bar(ax, labels, values, colors, fmt="${:.2f}", ylabel="",
                title="", xlabel="", rotation=0, val_offset_pct=0.02):
    """Draw vertical bars with glow shadows, value labels, and the
    project's dark style applied."""
    bars = ax.bar(
        labels, values,
        color=colors, width=0.50,
        edgecolor="white", linewidth=0.3,
        zorder=3,
    )
    # glow shadow
    for bar, c in zip(bars, colors):
        ax.bar(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            width=bar.get_width() * 1.20,
            color=c, alpha=0.08, zorder=2,
        )
    # value labels
    ymax = max(values)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + ymax * val_offset_pct,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        )
    ax.set_ylim(0, ymax * 1.18)
    if rotation:
        ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=10)
    return bars


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("Loading feature store …")
    df = pd.read_parquet(DATA_PATH)

    # pre-compute helpers ──────────────────────────────────────────────
    def _temp_band(row):
        if row["temp_bin_cold"] == 1:
            return "Cold"
        elif row["temp_bin_hot"] == 1:
            return "Hot"
        return "Moderate"

    df["temp_band"] = df.apply(_temp_band, axis=1)
    df["hour"] = pd.to_datetime(df["time_bucket"]).dt.hour

    # ──────────────────────────────────────────────────────────────────
    # 1.  Trip Count vs Rain
    # ──────────────────────────────────────────────────────────────────
    grp = df.groupby("is_raining", as_index=False)["trip_count"].mean()
    labels = ["No Rain", "Rain"]
    vals   = [grp.loc[grp["is_raining"] == 0, "trip_count"].values[0],
              grp.loc[grp["is_raining"] == 1, "trip_count"].values[0]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = _styled_bar(ax, labels, vals, WEATHER_PAIR,
                       fmt="{:.1f}", ylabel="Average Trip Count",
                       title="Average Trip Count: Rain vs No Rain")
    # highlight the larger bar
    max_idx = int(np.argmax(vals))
    bars[max_idx].set_edgecolor(C_AMBER)
    bars[max_idx].set_linewidth(1.5)
    # delta annotation
    pct_change = (vals[1] - vals[0]) / vals[0] * 100
    sign = "+" if pct_change > 0 else ""
    ax.text(
        0.98, 0.95,
        f"Δ  {sign}{pct_change:.1f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=13, fontweight="bold",
        color=C_AMBER, fontfamily=FONT_FAMILY,
    )
    apply_dark_style(ax, fig, title="Average Trip Count: Rain vs No Rain",
                     xlabel="Weather Condition", ylabel="Average Trip Count")
    save_fig("trip_count_vs_rain")
    print("  ✓ trip_count_vs_rain.png")

    # ──────────────────────────────────────────────────────────────────
    # 2.  Fare per Mile: Dry vs Wet
    # ──────────────────────────────────────────────────────────────────
    grp = df.groupby("is_wet_weather", as_index=False)["med_fare_per_mile"].mean()
    labels = ["Dry", "Wet Weather"]
    vals   = [grp.loc[grp["is_wet_weather"] == 0, "med_fare_per_mile"].values[0],
              grp.loc[grp["is_wet_weather"] == 1, "med_fare_per_mile"].values[0]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = _styled_bar(ax, labels, vals, WEATHER_PAIR,
                       fmt="${:.2f}", ylabel="Average Median Fare / Mile",
                       title="Fare per Mile: Dry vs Wet Weather")
    max_idx = int(np.argmax(vals))
    bars[max_idx].set_edgecolor(C_AMBER)
    bars[max_idx].set_linewidth(1.5)
    pct_change = (vals[1] - vals[0]) / vals[0] * 100
    sign = "+" if pct_change > 0 else ""
    ax.text(
        0.98, 0.95,
        f"Δ  {sign}{pct_change:.1f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=13, fontweight="bold",
        color=C_AMBER, fontfamily=FONT_FAMILY,
    )
    apply_dark_style(ax, fig, title="Fare per Mile: Dry vs Wet Weather",
                     xlabel="Weather Condition",
                     ylabel="Average Median Fare / Mile ($)")
    save_fig("fare_vs_wet_weather")
    print("  ✓ fare_vs_wet_weather.png")

    # ──────────────────────────────────────────────────────────────────
    # 3.  Pickup Delay vs Rain
    # ──────────────────────────────────────────────────────────────────
    grp = df.groupby("is_raining", as_index=False)["avg_pickup_delay_sec"].mean()
    labels = ["No Rain", "Rain"]
    vals   = [grp.loc[grp["is_raining"] == 0, "avg_pickup_delay_sec"].values[0],
              grp.loc[grp["is_raining"] == 1, "avg_pickup_delay_sec"].values[0]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = _styled_bar(ax, labels, vals, WEATHER_PAIR,
                       fmt="{:.1f}s", ylabel="Average Pickup Delay (sec)")
    max_idx = int(np.argmax(vals))
    bars[max_idx].set_edgecolor(C_AMBER)
    bars[max_idx].set_linewidth(1.5)
    pct_change = (vals[1] - vals[0]) / vals[0] * 100
    sign = "+" if pct_change > 0 else ""
    ax.text(
        0.98, 0.95,
        f"Δ  {sign}{pct_change:.1f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=13, fontweight="bold",
        color=C_AMBER, fontfamily=FONT_FAMILY,
    )
    apply_dark_style(ax, fig, title="Pickup Delay: Rain vs No Rain",
                     xlabel="Weather Condition",
                     ylabel="Average Pickup Delay (sec)")
    save_fig("delay_vs_rain")
    print("  ✓ delay_vs_rain.png")

    # ──────────────────────────────────────────────────────────────────
    # 4.  Trip Count by Temperature Band
    # ──────────────────────────────────────────────────────────────────
    grp = df.groupby("temp_band", as_index=False)["trip_count"].mean()
    order = ["Cold", "Moderate", "Hot"]
    grp["temp_band"] = pd.Categorical(grp["temp_band"], categories=order, ordered=True)
    grp = grp.sort_values("temp_band")
    labels = grp["temp_band"].astype(str).values
    vals   = grp["trip_count"].values

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = _styled_bar(ax, labels, vals, TEMP_TRIO,
                       fmt="{:.1f}", ylabel="Average Trip Count")
    max_idx = int(np.argmax(vals))
    bars[max_idx].set_edgecolor(C_AMBER)
    bars[max_idx].set_linewidth(1.5)
    apply_dark_style(ax, fig, title="Trip Count by Temperature Band",
                     xlabel="Temperature Band",
                     ylabel="Average Trip Count")
    save_fig("trip_count_vs_temperature")
    print("  ✓ trip_count_vs_temperature.png")

    # ──────────────────────────────────────────────────────────────────
    # 5.  Fare Volatility: Dry vs Wet
    # ──────────────────────────────────────────────────────────────────
    grp = df.groupby("is_wet_weather", as_index=False)["med_fare_per_mile"].std()
    labels = ["Dry", "Wet Weather"]
    vals   = [grp.loc[grp["is_wet_weather"] == 0, "med_fare_per_mile"].values[0],
              grp.loc[grp["is_wet_weather"] == 1, "med_fare_per_mile"].values[0]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = _styled_bar(ax, labels, vals, WEATHER_PAIR,
                       fmt="${:.3f}", ylabel="Std Dev of Fare / Mile ($)")
    max_idx = int(np.argmax(vals))
    bars[max_idx].set_edgecolor(C_AMBER)
    bars[max_idx].set_linewidth(1.5)
    # WSR annotation
    wsr = vals[1] / vals[0] if vals[0] != 0 else float("inf")
    ax.text(
        0.98, 0.95,
        f"WSR = {wsr:.2f}×",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=13, fontweight="bold",
        color=C_AMBER, fontfamily=FONT_FAMILY,
    )
    apply_dark_style(ax, fig,
                     title="Price Volatility: Dry vs Wet Weather",
                     xlabel="Weather Condition",
                     ylabel="Std Dev of Fare per Mile ($)")
    save_fig("volatility_vs_weather")
    print("  ✓ volatility_vs_weather.png")

    # ──────────────────────────────────────────────────────────────────
    # 6.  Hourly Demand by Rain Condition  (dual glow-line chart)
    # ──────────────────────────────────────────────────────────────────
    hourly = df.groupby(["hour", "is_raining"])["trip_count"].mean().unstack()

    x = hourly.index.values.astype(float)
    y_no_rain = hourly[0].values
    y_rain    = hourly[1].values

    x_smooth = np.linspace(x.min(), x.max(), 300)

    fig, ax = plt.subplots(figsize=(11, 6))

    y_all = np.concatenate([y_no_rain, y_rain])
    y_pad = (y_all.max() - y_all.min()) * 0.15
    y_min = y_all.min() - y_pad
    y_max = y_all.max() + y_pad
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, 23.5)

    # No-rain line (cyan)
    spl0 = make_interp_spline(x, y_no_rain, k=3)
    y0s  = spl0(x_smooth)
    add_gradient_fill(ax, x_smooth, y0s, C_CYAN, C_PURPLE, alpha=0.15, y_min=y_min)
    add_glow_line(ax, x_smooth, y0s, color=C_CYAN, linewidth=2.5, label="No Rain")
    ax.scatter(x, y_no_rain, color=C_CYAN, s=28, zorder=6,
               edgecolors="white", linewidth=0.5)

    # Rain line (pink)
    spl1 = make_interp_spline(x, y_rain, k=3)
    y1s  = spl1(x_smooth)
    add_gradient_fill(ax, x_smooth, y1s, C_PINK, C_PURPLE, alpha=0.12, y_min=y_min)
    add_glow_line(ax, x_smooth, y1s, color=C_PINK, linewidth=2.5, label="Rain")
    ax.scatter(x, y_rain, color=C_PINK, s=28, zorder=6,
               edgecolors="white", linewidth=0.5)

    # Peak annotations – No Rain
    peak0 = int(np.argmax(y_no_rain))
    ax.annotate(
        f"Peak: {y_no_rain[peak0]:.1f}",
        xy=(x[peak0], y_no_rain[peak0]),
        xytext=(x[peak0] + 2.5, y_no_rain[peak0] + y_pad * 0.4),
        fontsize=10, fontweight="bold",
        color=C_CYAN, fontfamily=FONT_FAMILY,
        arrowprops=dict(arrowstyle="->", color=C_CYAN, lw=1.5),
        zorder=7,
    )
    # Peak annotations – Rain
    peak1 = int(np.argmax(y_rain))
    ax.annotate(
        f"Peak: {y_rain[peak1]:.1f}",
        xy=(x[peak1], y_rain[peak1]),
        xytext=(x[peak1] - 3.5, y_rain[peak1] + y_pad * 0.6),
        fontsize=10, fontweight="bold",
        color=C_PINK, fontfamily=FONT_FAMILY,
        arrowprops=dict(arrowstyle="->", color=C_PINK, lw=1.5),
        zorder=7,
    )

    apply_dark_style(ax, fig,
                     title="Average Trip Count by Hour & Rain Condition",
                     xlabel="Hour of Day",
                     ylabel="Average Trip Count")
    # AM/PM labels
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

    # legend
    legend_elements = [
        Patch(facecolor=C_CYAN, label="No Rain"),
        Patch(facecolor=C_PINK, label="Rain"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=10, framealpha=0.6,
        facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        bbox_to_anchor=(1.0, 1.12), ncol=2,
    )
    save_fig("hourly_demand_weather")
    print("  ✓ hourly_demand_weather.png")

    # ──────────────────────────────────────────────────────────────────
    # 7.  Top 10 Zones Most Affected by Wet Weather
    # ──────────────────────────────────────────────────────────────────
    zone_weather = (
        df.groupby(["PULocationID", "is_wet_weather"])["med_fare_per_mile"]
        .mean()
        .unstack()
        .fillna(0)
    )
    zone_weather["wet_minus_dry"] = zone_weather[1] - zone_weather[0]
    top_zones = zone_weather.sort_values("wet_minus_dry", ascending=False).head(10)
    top_zones = top_zones.sort_values("wet_minus_dry", ascending=True)  # low→high for barh

    # Map zone IDs → human-readable names via lookup table
    zones_csv = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "taxi_zone_lookup.csv")
    zone_name_map = dict(zip(zones_csv["LocationID"], zones_csv["Zone"]))
    zone_labels = [zone_name_map.get(int(z), f"Zone {int(z)}") for z in top_zones.index]
    vals = top_zones["wet_minus_dry"].values

    fig, ax = plt.subplots(figsize=(11, 7))

    norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())
    colors = [NEON_CMAP(norm(v)) for v in vals]

    bars = ax.barh(
        range(len(top_zones)), vals,
        color=colors, height=0.60,
        edgecolor="white", linewidth=0.3,
        zorder=3,
    )
    # glow behind bars
    for bar, c in zip(bars, colors):
        ax.barh(
            bar.get_y() + bar.get_height() / 2,
            bar.get_width(),
            height=bar.get_height() * 1.2,
            color=c, alpha=0.08, zorder=2,
        )
    # value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + vals.max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"+${w:.2f}",
            va="center", ha="left",
            fontsize=10, fontweight="bold",
            color=TEXT_COLOR, fontfamily=FONT_FAMILY,
        )
    # highlight top zone
    bars[-1].set_edgecolor(C_AMBER)
    bars[-1].set_linewidth(1.5)

    apply_dark_style(ax, fig,
                     title="Top 10 Zones Most Affected by Wet Weather",
                     xlabel="Fare Increase: Wet − Dry ($/mile)",
                     ylabel="Pickup Zone")
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.4, alpha=0.5)
    ax.grid(False, axis="y")
    ax.set_yticks(range(len(top_zones)))
    ax.set_yticklabels(zone_labels)
    ax.set_xlim(0, vals.max() * 1.18)
    save_fig("top_zones_weather_impact")
    print("  ✓ top_zones_weather_impact.png")

    # ──────────────────────────────────────────────────────────────────
    #  Summary table + Weather Stress Ratio
    # ──────────────────────────────────────────────────────────────────
    dry_vol = df[df["is_wet_weather"] == 0]["med_fare_per_mile"].std()
    wet_vol = df[df["is_wet_weather"] == 1]["med_fare_per_mile"].std()

    summary = {
        "avg_trip_no_rain":     df[df["is_raining"] == 0]["trip_count"].mean(),
        "avg_trip_rain":        df[df["is_raining"] == 1]["trip_count"].mean(),
        "avg_fare_dry":         df[df["is_wet_weather"] == 0]["med_fare_per_mile"].mean(),
        "avg_fare_wet":         df[df["is_wet_weather"] == 1]["med_fare_per_mile"].mean(),
        "avg_delay_no_rain":    df[df["is_raining"] == 0]["avg_pickup_delay_sec"].mean(),
        "avg_delay_rain":       df[df["is_raining"] == 1]["avg_pickup_delay_sec"].mean(),
        "avg_trip_cold":        df[df["temp_bin_cold"] == 1]["trip_count"].mean(),
        "avg_trip_not_cold":    df[df["temp_bin_cold"] == 0]["trip_count"].mean(),
        "volatility_dry":       dry_vol,
        "volatility_wet":       wet_vol,
        "weather_stress_ratio": wet_vol / dry_vol,
    }

    summary_df = pd.DataFrame([summary])
    summary_path = EDA_DIR / "eda_summary_table.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n📊 SUMMARY TABLE:\n{summary_df}")
    print(f"\n✅ All charts saved to: {FIGURES_DIR}")
    print(f"✅ Summary CSV saved to: {summary_path}")


if __name__ == "__main__":
    main()