# Heteroscedasticity Summary (Fare Prediction)

## Goal
Check whether prediction errors vary across the fare range (heteroscedasticity) and evaluate fixes.

## Diagnostic Method
We computed residual variability by prediction decile and plotted residual standard deviation vs. decile.

**Script:** `scripts/eval_slices.py` with `--hetero` flag

**Outputs:**
- `reports/slices/hetero_resid_by_decile_YYYY-MM.csv`
- `reports/slices/hetero_resid_by_decile_YYYY-MM.png`

## Baseline (Raw Target)
**Result:** Residual variance increases with predicted fare.
- Low decile residual std dev ≈ 0.22
- Top decile residual std dev ≈ 2.33

**Conclusion:** Strong heteroscedasticity in baseline model.

## Fix Attempt 1: Log-Transform Target
**Change:** Train on `log1p(base_passenger_fare)` and invert predictions with `expm1` for metrics.

**Result:** Similar residual spread to baseline.
- Low decile residual std dev ≈ 0.21
- Top decile residual std dev ≈ 2.51

**Conclusion:** Log transform did not materially reduce heteroscedasticity.

## Fix Attempt 2: Segment-Specific Models
**Change:** Train separate models for `short`, `medium`, and `long` trips.

**Results:**
- **Short:** residual std dev ~0.21 → ~2.28
- **Medium:** residual std dev ~0.17–0.26 → ~2.36
- **Long:** residual std dev ~0.78–1.2 → ~3.79

**Conclusion:** Segmentation helps slightly in low/mid deciles but does not fix top-decile variance, especially for long trips.

## Fix Attempt 3: Quantile Loss (Median Regression)
**Change:** `HistGradientBoostingRegressor(loss="quantile", quantile=0.5)`

**Result:** Heteroscedasticity remained; top decile residual std dev ≈ 3.74.

**Conclusion:** Quantile loss did not reduce variance growth at the high end.

## Overall Conclusion
Heteroscedasticity persists, primarily driven by high-fare trips. Log-transform, segmentation, and quantile loss did not fully resolve it.



### Commands Used
```bash
# Baseline diagnostics
python scripts/eval_slices.py --test_month 2021-10 --sample_n 0 --hetero

# Log-transform training
a
python scripts/train_model.py --log_target --output_dir models/final_model_log
python scripts/eval_slices.py --test_month 2021-10 --sample_n 0 --hetero --model_dir models/final_model_log

# Segment training
python scripts/train_model.py --segment_by_trip_length --output_dir models/final_model_segmented

# Segment diagnostics (example)
python scripts/eval_slices.py --test_month 2021-10 --sample_n 0 --hetero --segment long \
  --model_dir models/final_model_segmented/segment=long

# Quantile loss training
python scripts/train_model.py --loss quantile --quantile 0.5 --output_dir models/final_model_quantile
python scripts/eval_slices.py --test_month 2021-10 --sample_n 0 --hetero --model_dir models/final_model_quantile
```
