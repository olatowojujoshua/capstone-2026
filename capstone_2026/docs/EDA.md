# Exploratory Data Analysis (EDA)

## 1. Purpose of EDA

The primary objective of this study is to examine fare stability, spatial fairness, and pricing volatility within high-volume for-hire vehicle trip data.

Exploratory Data Analysis (EDA) serves as a structured, hypothesis-driven investigation designed to:

- Characterize the structure, scale, and integrity of the dataset
- Identify temporal pricing patterns and demand-driven variability
- Quantify fare volatility across time and geography
- Analyze trip-level characteristics influencing pricing outcomes
- Detect potential fairness-related signals embedded in pricing behavior

The EDA framework is intentionally designed to be:
- Memory-efficient through aggregation-based computation
- Reproducible and methodologically transparent
- Aligned with fairness and stability hypotheses

---

## 2. Dataset Overview

The dataset consists of high-volume trip-level records, including:

- Pickup and dropoff timestamps
- Pickup and dropoff zones
- Trip distance (`trip_miles`)
- Base passenger fare (`base_passenger_fare`)
- Additional fare components (taxes, congestion surcharge, tolls, airport fees, etc.)

With over 100 million observations, the scale of the dataset necessitates aggregated analytical strategies rather than raw-level visualization to ensure computational feasibility and statistical robustness.

---

## 3. Temporal Analysis

### Rationale

Dynamic pricing systems adjust fares in response to fluctuations in demand over time.  
Therefore, temporal analysis is critical for assessing:

- Surge pricing behavior
- Stability across hourly and daily demand cycles
- Temporal fairness in fare distribution

### Analytical Focus

The temporal analysis includes:

- Average fare by hour of day
- Hourly fare standard deviation as a volatility measure
- Smoothed temporal volatility trends

These measures facilitate identification of peak instability intervals and structural changes in pricing behavior over time.

---

## 4. Spatial Analysis

### Rationale

Pricing fairness must be evaluated across geographic regions to detect potential disparities.  
Persistent differences in mean fare or volatility across zones may signal spatial inequities.

### Analytical Focus

Spatial evaluation includes:

- Mean fare by pickup zone
- Standard deviation of fares by zone
- Trip density (trip counts) by zone

This analysis enables assessment of geographic pricing heterogeneity and spatial fairness patterns.

---

## 5. Trip-Level Characteristics

### Rationale

Trip distance plays a structural role in fare formation.  
Short-distance trips may exhibit disproportionately high per-mile costs, potentially indicating regressive pricing effects.

### Analytical Focus

- Fare per mile computation
- Categorization into trip length buckets (short, medium, long)
- Average fare across distance categories

This supports investigation of structural pricing bias related to trip characteristics.

---

## 6. Fare Component Decomposition

### Rationale

Pricing transparency and fairness are influenced not only by total fare but also by the composition of fare components.  
Understanding the contribution of each component aids in interpreting variability and volatility.

### Analytical Focus

- Base passenger fare
- Sales tax
- Congestion surcharge
- Airport fees
- Tolls
- Driver compensation

Component-level analysis clarifies which elements drive overall fare variability and instability.

---

## 7. Volatility and Stability Assessment

### Rationale

Fare volatility serves as a quantitative proxy for pricing instability.  
Excessive variability may reduce predictability, weaken consumer trust, and raise fairness concerns.

### Analytical Focus

- Hourly fare standard deviation
- Volatility patterns across hours of the day
- Smoothed long-term volatility trends

This analysis directly supports evaluation of dynamic pricing stability.

---

## 8. Execution Instructions

All EDA modules are orchestrated through a centralized execution script to ensure reproducibility and consistency across analytical outputs.

To execute the complete EDA pipeline, run the following command from the project root directory:

```bash
python main_eda.py