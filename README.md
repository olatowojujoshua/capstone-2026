# Ride Fare Pricing, Volatility, and Fairness Analysis  
## Capstone Project – 2026

---

## 1. Project Overview

This project analyzes ride-hailing fare dynamics using high-volume trip-level data.  
The study investigates pricing behavior, temporal volatility, and fairness across geographic zones.

The project implements a structured data science pipeline:

**Exploratory Data Analysis → Feature Engineering → Modeling → Evaluation → Fairness Assessment**

The objective is to build reproducible machine learning models while examining pricing stability and equity.

---

## 2. Research Objectives

This project addresses the following objectives:

1. **Fare Prediction**
   - Model base passenger fare using trip-level features.
   - Evaluate predictive performance using multiple regression metrics.

2. **Volatility Modeling**
   - Model hourly fare variability.
   - Identify temporal patterns in pricing instability.

3. **Fairness Evaluation**
   - Assess model error across pickup zones.
   - Evaluate prediction disparities across hours of the day.
   - Quantify tail-risk using high-percentile error metrics.

---

## 3. Dataset Description

The dataset consists of high-volume ride trip records stored as monthly parquet files.

Location:
```
data/interim/YYYY-MM_clean.parquet
```

Key variables include:

| Column | Description |
|--------|------------|
| `pickup_datetime` | Pickup timestamp |
| `dropoff_datetime` | Dropoff timestamp |
| `PULocationID` | Pickup zone ID |
| `DOLocationID` | Dropoff zone ID |
| `trip_miles` | Trip distance |
| `trip_time` | Trip duration |
| `base_passenger_fare` | Base fare (target variable) |
| `driver_pay` | Driver compensation |
| `fare_per_mile` | Engineered feature |

---

## 4. Project Structure

```
capstone_2026/
│
├── data/                               # Monthly cleaned parquet files
|
├── docs/                               # All code explanations (.md files)
│
├── models/                             # Save trained models
│
├── reports/                            # Save output reports
│
├── src/                                # Code workflows and pipelines
│
├── main_model_building.py              # To run model building workflow
├── main_eda.py                         # To create EDA and save it in reports directory
├── main_model_building.py              # To run model building workflow
└── main_plot.py                        # To visualize and create all plots
```

---

## 5. Design Principles

This project follows:

- Modular architecture  
- Shared preprocessing pipeline  
- Idempotent training design  
- Structured evaluation outputs  
- Reproducible modeling workflow  

---

## 6. Future Extensions

Potential improvements include:

- Time-series forecasting for volatility  
- SHAP-based model explainability  
- Bias disparity index calculation  
- Temporal cross-validation  
- Deployment-ready inference pipeline  

---

## 7. Author

Group 7  
Master of Data Analytics – Capstone 2026  

---

## 8. License

This project is developed for academic purposes.