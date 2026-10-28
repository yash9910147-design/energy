# Experiment 4: EDA Insights

## Findings

1. Usage_kWh has skewness 1.20; the distribution is right-skewed relative to a symmetric distribution.
2. Mean usage is 27.39 kWh per interval and median is 4.57 kWh, indicating that the median is a useful robust reference.
3. Weekend mean usage is 11.73 kWh and weekday mean usage is 33.62 kWh.
4. Maximum_Load has the highest average usage (59.27 kWh per interval) among the recorded load types.
5. Monthly average usage is highest in January (42.42 kWh) and lowest in December (19.97 kWh); one year cannot establish a recurring annual season.
6. `CO2(tCO2)` has the largest absolute Pearson correlation with Usage_kWh among the numeric source features (r=0.988).
7. Autocorrelation is 0.912 at 15 minutes and 0.606 at 24 hours, supporting chronological lag features.
8. About 0.9% of target readings lie outside the 1.5-IQR whiskers; inspect these operating peaks rather than deleting them automatically.

## Implications for modeling

- Preserve chronological ordering and compare against a persistence baseline.
- Include calendar and lag features; fit all imputers/encoders on training folds only.
- Keep `Load_Type` and power-factor readings only when available at forecast time.
- Exclude `CO2(tCO2)` from predictors until its timing is proven; its strong target correlation could reveal same-interval energy use.
- Retain operational peaks and evaluate errors by load type and time period.
