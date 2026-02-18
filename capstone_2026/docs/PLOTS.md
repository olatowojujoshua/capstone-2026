# Visualization & Plotting Strategy

## 1. Purpose of Visualizations

Visualizations serve as structured analytical tools designed to:

- Communicate pricing patterns clearly and systematically
- Support hypothesis evaluation
- Provide interpretable evidence of fairness and stability
- Prevent distortion caused by raw-level over-plotting

All figures are generated from aggregated EDA outputs to ensure computational scalability, interpretability, and methodological consistency.

---

## 2. Temporal Visualizations

### Average Fare by Hour of Day

**Purpose**

- Identify peak pricing periods  
- Detect systematic temporal pricing patterns  
- Evaluate time-based fairness dynamics  

**Analytical Insight**

- Are fares consistently elevated during specific hours of the day?

---

### Hourly Fare Volatility (Trend Over Time)

**Purpose**

- Measure pricing stability across extended time horizons  
- Detect long-term structural instability  
- Identify temporal regime shifts in pricing behavior  

**Analytical Insight**

- Is pricing volatility increasing or decreasing over time?

---

### Average Volatility by Hour of Day

**Purpose**

- Identify periods of peak pricing instability  
- Evaluate surge intensity during high-demand intervals  

**Analytical Insight**

- Do certain hours systematically exhibit higher unpredictability?

---

## 3. Spatial Visualizations

### Average Fare by Pickup Zone

**Purpose**

- Evaluate geographic pricing disparities  
- Identify consistently high-cost zones  

**Analytical Insight**

- Are certain areas systematically more expensive than others?

---

## 4. Trip-Based Visualizations

### Fare by Trip Length Category

**Purpose**

- Evaluate potential structural bias against short-distance trips  
- Compare pricing levels across trip-length categories  

**Analytical Insight**

- Are short trips disproportionately expensive on a per-mile basis?

---

## 5. Fare Component Breakdown

**Purpose**

- Improve transparency in fare composition  
- Identify primary contributors to total fare variability  

**Analytical Insight**

- Do surcharges or auxiliary fees significantly drive volatility?

---

## 6. Visualization Design Principles

All visual outputs adhere to the following methodological principles:

- Aggregated data only (no raw-level plotting)
- One primary analytical insight per figure
- Clear axis labeling and interpretable scales
- Minimal stylistic distortion (no excessive coloring or embellishment)
- Exported as reproducible PNG files for reporting consistency

These principles ensure clarity, neutrality, and analytical integrity.

---

## 7. Execution Instructions

All visualization modules are orchestrated through a centralized plotting pipeline to ensure reproducibility and consistency.

To generate all visual outputs, execute the following command from the project root directory:

```bash
python main_plot.py
