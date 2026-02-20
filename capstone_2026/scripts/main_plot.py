from src.eda_plots import (
    plot_temporal,
    plot_spatial,
    plot_trip_length,
    plot_fare_breakdown,
    plot_volatility_by_hour_of_day,
    plot_volatility_trend_over_time
)

def main():
    plot_temporal.run()
    plot_spatial.run()
    plot_trip_length.run()
    plot_fare_breakdown.run()
    plot_volatility_by_hour_of_day.run()
    plot_volatility_trend_over_time.run()

if __name__ == "__main__":
    main()