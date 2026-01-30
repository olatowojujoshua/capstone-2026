# capstone-2026
# Data Quality and Preprocessing Assessment

## Dataset Overview
This project uses publicly available New York City High-Volume For-Hire Vehicle (HVFHS) trip records released by the NYC Taxi and Limousine Commission (TLC). The dataset contains trip-level observations for high-volume ride-hailing platforms including Uber (HV0003), Lyft (HV0005), Via (HV0004), and Juno (HV0002).

The analysis covers **January–December 2021**, comprising approximately **178 million completed trips**. Each record represents an observed ride outcome and includes trip distance and duration, pickup and drop-off zones, request and pickup timestamps, platform identifiers, and a detailed breakdown of passenger fares and driver pay.

The dataset is treated as **observational administrative data**. No pricing rules are modified or simulated; all analyses are based strictly on observed fares.

---

## Data Ingestion Strategy
Due to the large size of the dataset (~4.7 GB across 12 months), raw data are processed **one month at a time** using a script-based pipeline. This design ensures:
- Memory-efficient execution
- Temporal traceability
- Reproducibility across environments (VS Code, Google Colab)

Each raw monthly file is cleaned independently and saved as a compressed Parquet file in the interim data layer.

---

## Cleaning Rules and Validation Checks

The following preprocessing rules were applied consistently across all months:

### 1. Column Validation
Only expected HVFHS columns were retained. The pipeline safely handled minor schema variations across months.

### 2. Datetime Normalization
Request, pickup, and drop-off timestamps were converted to datetime format with invalid values coerced to missing.

### 3. Critical Field Completeness
Trips missing any of the following fields were excluded:
- Pickup timestamp  
- Pickup and drop-off zone IDs  
- Trip distance  
- Trip duration  
- Base passenger fare  

**Result:** No records were dropped due to missing critical fields in any month.

### 4. Invalid Trip Removal
Trips with non-positive values for:
- Trip distance  
- Trip duration  
- Base passenger fare  

were removed, as these represent invalid or corrupted records.

### 5. Pickup Delay Sanity Filter
Pickup delay (request → pickup) was computed as a proxy for demand–supply pressure. Delays outside the range **0–2 hours** were excluded to remove logging artifacts.

### 6. Outlier Control (Winsorization)
To limit the influence of extreme values without distorting distributions, conservative winsorization thresholds were identified at the **0.5th and 99.5th percentiles** for:
- Trip distance  
- Trip duration  
- Base passenger fare  
- Fare per mile  

These thresholds were reported and applied consistently during modeling.

---

## Data Quality Results (January–December 2021)
processed 178 million trips

### Record Retention
Across all months:
- Between **0.27% and 1.87%** of records were removed
- No month exceeded **2% total removal**
- The majority of removals were due to extreme pickup delays or invalid fare values

This indicates **high completeness and reliability** of the underlying data.

### Pickup Delay Behavior
Average pickup delays ranged from approximately **4 to 6 minutes**, consistent with expected ride-hailing behavior in dense urban environments. Higher out-of-range delays observed in late summer and fall months were treated as logging anomalies and excluded.

### Stability of Distributions
Winsorization thresholds evolved smoothly across months, reflecting seasonal demand changes and congestion patterns rather than data corruption. No abrupt shifts in trip length, duration, or fare distributions were observed.

---

## Interpretation and Implications

Overall, the HVFHS dataset demonstrates **exceptionally high data quality** for large-scale administrative records. Preprocessing required minimal correction and did not materially alter the underlying distribution of trips or fares.

This supports the validity of subsequent analyses, including:
- Fare predictability modeling  
- Short-horizon price stability analysis  
- Spatial and temporal fairness evaluation  

The conservative cleaning approach ensures that results reflect **true observed pricing behavior** rather than artifacts of preprocessing.

---

## Reproducibility
All data quality checks and preprocessing steps are fully reproducible using script-based workflows. A per-month data quality report (`data_quality_report.csv`) is generated to document row counts, removal reasons, and outlier thresholds, ensuring transparency and auditability.
