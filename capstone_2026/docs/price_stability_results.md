# Price Stability Analysis (Observed Fare Behavior)

## Objective
To evaluate short-horizon price volatility in observed ride-hailing fares using 15-minute time buckets, without modifying or simulating pricing rules.

## Method
Observed base passenger fares were aggregated into 15-minute time windows across all platforms and zones. For each bucket, mean fares were computed and compared to the immediately preceding time bucket to quantify absolute and percentage price changes. Buckets with insufficient trip volume were excluded to reduce noise.

## Key Findings
Across all months analyzed, observed fare changes between adjacent 15-minute windows were generally small. Average absolute price changes ranged between approximately $0.23 and $0.35, while the 90th percentile of absolute price changes remained below $1.10 in most months. Percentage price changes averaged approximately 1–1.5%, with the 90th percentile remaining below 5%.

Extreme price shocks were rare. Price jumps exceeding 25% occurred in fewer than 0.02% of observed time buckets, and price jumps exceeding 50% were virtually nonexistent.

## Interpretation
These results indicate that, at a 15-minute temporal resolution, observed dynamic pricing behavior in NYC ride-hailing data is generally stable, with limited short-term volatility and infrequent extreme price changes. This empirical stability motivates further analysis of whether pricing behavior is equally consistent across geographic zones and platforms.
