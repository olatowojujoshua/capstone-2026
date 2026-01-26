# Baseline Fare Model — Results (Observed Pricing Evaluation)

## Objective
Build a baseline model to predict **observed ride fares** using trip fundamentals and basic spatio-temporal context.  
This model is used to **evaluate fare predictability and error behavior** (including tail risk), not to recommend new prices.

## Data Window (Development Split)
- Train: 2021-01 to 2021-08
- Validation: 2021-09
- Test: 2021-10
- Dataset: NYC TLC HVFHS trip records (cleaned monthly parquet)

## Target Variable
- base_passenger_fare (observed)

## Feature Inputs
- Trip fundamentals: trip_miles, trip_time
- Spatio-temporal: hour, day-of-week, month
- Location: PULocationID, DOLocationID
- Platform: hvfhs_license_num

## Model
- HistGradientBoostingRegressor
- Categorical encoding: ordinal encoding (memory-safe)

## Performance Metrics

### Validation (2021-09)
- MAE: $4.73
- RMSE: $7.93
- WAPE: 19.93%
- P90 Absolute Error: $10.33

### Test (2021-10)
- MAE: $4.37
- RMSE: $6.76
- WAPE: 19.52%
- P90 Absolute Error: $9.15

## Interpretation (Analytics Framing)
The baseline model shows that observed fares are reasonably predictable from trip fundamentals and spatio-temporal context, but the tail error (P90 absolute error ≈ $9–$10) indicates meaningful variance in pricing outcomes beyond what trip fundamentals alone explain. This motivates further analysis of **pricing stability and fairness** across time windows, zones, and platforms using observational metrics rather than price optimization.
