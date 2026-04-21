import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== PATHS =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = r"C:\Users\olato\Downloads\group 7 capstone\local_capstone_2026\data\processed\zone_time_features\zone_time_features.parquet"

# ===== LOAD DATA =====
df = pd.read_parquet(DATA_PATH)

# (Optional) sample to speed things up
# df = df.sample(200000, random_state=42)

# ================================
# 1. Trip Count vs Rain
# ================================
plot_df = df.groupby("is_raining", as_index=False)["trip_count"].mean()
plot_df["is_raining"] = plot_df["is_raining"].map({0: "No Rain", 1: "Rain"})

plt.figure()
plt.bar(plot_df["is_raining"], plot_df["trip_count"])
plt.title("Average Trip Count: Rain vs No Rain")
plt.xlabel("Weather Condition")
plt.ylabel("Average Trip Count")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "trip_count_vs_rain.png")
plt.close()

# ================================
# 2. Fare vs Wet Weather
# ================================
plot_df = df.groupby("is_wet_weather", as_index=False)["med_fare_per_mile"].mean()
plot_df["is_wet_weather"] = plot_df["is_wet_weather"].map({0: "Dry", 1: "Wet Weather"})

plt.figure()
plt.bar(plot_df["is_wet_weather"], plot_df["med_fare_per_mile"])
plt.title("Fare per Mile: Dry vs Wet Weather")
plt.xlabel("Weather Condition")
plt.ylabel("Average Median Fare per Mile")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "fare_vs_wet_weather.png")
plt.close()

# ================================
# 3. Pickup Delay vs Rain
# ================================
plot_df = df.groupby("is_raining", as_index=False)["avg_pickup_delay_sec"].mean()
plot_df["is_raining"] = plot_df["is_raining"].map({0: "No Rain", 1: "Rain"})

plt.figure()
plt.bar(plot_df["is_raining"], plot_df["avg_pickup_delay_sec"])
plt.title("Pickup Delay: Rain vs No Rain")
plt.xlabel("Weather Condition")
plt.ylabel("Average Pickup Delay (sec)")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "delay_vs_rain.png")
plt.close()

# ================================
# 4. Trip Count vs Temperature Band
# ================================
def temp_band(row):
    if row["temp_bin_cold"] == 1:
        return "Cold"
    elif row["temp_bin_hot"] == 1:
        return "Hot"
    return "Moderate"

df["temp_band"] = df.apply(temp_band, axis=1)

plot_df = df.groupby("temp_band", as_index=False)["trip_count"].mean()
order = ["Cold", "Moderate", "Hot"]
plot_df["temp_band"] = pd.Categorical(plot_df["temp_band"], categories=order, ordered=True)
plot_df = plot_df.sort_values("temp_band")

plt.figure()
plt.bar(plot_df["temp_band"], plot_df["trip_count"])
plt.title("Trip Count by Temperature Band")
plt.xlabel("Temperature Band")
plt.ylabel("Average Trip Count")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "trip_count_vs_temperature.png")
plt.close()

plot_df = df.groupby("is_wet_weather", as_index=False)["med_fare_per_mile"].std()
plot_df["is_wet_weather"] = plot_df["is_wet_weather"].map({0: "Dry", 1: "Wet Weather"})

plt.figure()
plt.bar(plot_df["is_wet_weather"], plot_df["med_fare_per_mile"])
plt.title("Price Volatility: Dry vs Wet Weather")
plt.xlabel("Weather Condition")
plt.ylabel("Std Dev of Fare per Mile")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "volatility_vs_weather.png")
plt.close()

df["hour"] = pd.to_datetime(df["time_bucket"]).dt.hour

plot_df = df.groupby(["hour", "is_raining"])["trip_count"].mean().unstack()

plt.figure()
plot_df.plot()
plt.title("Trip Count by Hour and Rain Condition")
plt.xlabel("Hour of Day")
plt.ylabel("Average Trip Count")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "hourly_demand_weather.png")
plt.close()

df["hour"] = pd.to_datetime(df["time_bucket"]).dt.hour
plot_df = df.groupby(["hour", "is_raining"])["trip_count"].mean().unstack()

ax = plot_df.plot(figsize=(8, 5))
ax.set_title("Average Trip Count by Hour and Rain Condition")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average Trip Count")
ax.legend(["No Rain", "Rain"])
plt.tight_layout()
plt.savefig(FIGURES_DIR / "hourly_demand_weather.png")
plt.close()

# ================================
# 5. Fairness: Top Zones Most Affected by Weather
# ================================

zone_weather = df.groupby(["PULocationID", "is_wet_weather"])["med_fare_per_mile"].mean().unstack()

# Handle missing cases safely
zone_weather = zone_weather.fillna(0)

zone_weather["wet_minus_dry"] = zone_weather[1] - zone_weather[0]

top_zones = zone_weather.sort_values("wet_minus_dry", ascending=False).head(10)

plt.figure(figsize=(10, 5))
plt.bar(top_zones.index.astype(str), top_zones["wet_minus_dry"])
plt.title("Top 10 Zones Most Affected by Wet Weather")
plt.xlabel("PULocationID")
plt.ylabel("Fare Increase (Wet - Dry)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "top_zones_weather_impact.png")
plt.close()

print(f"✅ All charts saved to: {FIGURES_DIR}")

dry_vol = df[df["is_wet_weather"] == 0]["med_fare_per_mile"].std()
wet_vol = df[df["is_wet_weather"] == 1]["med_fare_per_mile"].std()

wsr = wet_vol / dry_vol

print(f"Weather Stress Ratio (WSR): {wsr:.2f}")

# ================================
# SUMMARY TABLE (ALL KEY INSIGHTS)
# ================================

summary = {}

# Demand
summary["avg_trip_no_rain"] = df[df["is_raining"] == 0]["trip_count"].mean()
summary["avg_trip_rain"] = df[df["is_raining"] == 1]["trip_count"].mean()

# Pricing
summary["avg_fare_dry"] = df[df["is_wet_weather"] == 0]["med_fare_per_mile"].mean()
summary["avg_fare_wet"] = df[df["is_wet_weather"] == 1]["med_fare_per_mile"].mean()

# Delay
summary["avg_delay_no_rain"] = df[df["is_raining"] == 0]["avg_pickup_delay_sec"].mean()
summary["avg_delay_rain"] = df[df["is_raining"] == 1]["avg_pickup_delay_sec"].mean()

# Temperature
summary["avg_trip_cold"] = df[df["temp_bin_cold"] == 1]["trip_count"].mean()
summary["avg_trip_not_cold"] = df[df["temp_bin_cold"] == 0]["trip_count"].mean()

# Volatility (VERY IMPORTANT)
dry_vol = df[df["is_wet_weather"] == 0]["med_fare_per_mile"].std()
wet_vol = df[df["is_wet_weather"] == 1]["med_fare_per_mile"].std()

summary["volatility_dry"] = dry_vol
summary["volatility_wet"] = wet_vol
summary["weather_stress_ratio"] = wet_vol / dry_vol

# Convert to DataFrame
summary_df = pd.DataFrame([summary])

# Save table
summary_path = FIGURES_DIR / "eda_summary_table.csv"
summary_df.to_csv(summary_path, index=False)

print("\n📊 SUMMARY TABLE:")
print(summary_df)
print(f"\n✅ Saved to: {summary_path}")