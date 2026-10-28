# Experiment 2: Dataset Description

- **Local source file used:** `Steel_industry_data 3.csv`.
- **Source:** UCI Machine Learning Repository, Steel Industry Energy Consumption, dataset 851, DOI: 10.24432/C52G8C.
- **Creators:** Sathishkumar V E, Changsun Shin, and Yongyun Cho.
- **Associated publication:** Efficient energy consumption prediction model for a data analytic-enabled industry building in a smart city, Building Research & Information, 49(1), 127-143 (2021).
- **Dataset page:** https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption
- **License:** The dataset 851 API metadata does not report a license. Verify the current UCI record's terms before redistribution; cite the dataset DOI.
- **Original shape:** 35,040 rows x 11 columns.
- **Cleaned time-indexed shape:** 35,040 rows x 11 columns.
- **Coverage:** 15-minute readings for calendar year 2018.
- **Regression target in this project:** `Usage_kWh` (continuous kWh). UCI's original published target is categorical `Load_Type`; this project reframes the data for energy regression and uses `Load_Type` as an input.

## Feature dictionary

| Column | Meaning |
| --- | --- |
| `date` | Local interval timestamp, recorded every 15 minutes. |
| `Usage_kWh` | Industry electricity consumption for the interval (project target). |
| `Lagging_Current_Reactive.Power_kVarh` | Lagging reactive energy measurement. |
| `Leading_Current_Reactive_Power_kVarh` | Leading reactive energy measurement. |
| `CO2(tCO2)` | CO2 field as named in the source dataset; preserve source units/definition. |
| `Lagging_Current_Power_Factor` | Lagging current power factor (percent in source metadata). |
| `Leading_Current_Power_Factor` | Leading current power factor (percent in source metadata). |
| `NSM` | Number of seconds from midnight. |
| `WeekStatus` | `Weekday` or `Weekend`. |
| `Day_of_week` | Day name, Sunday through Saturday. |
| `Load_Type` | Source load category: light, medium, or maximum load. |

## Data types and missing values

| column | dtype | missing_values |
| --- | --- | --- |
| date | datetime64[ns] | 0 |
| Usage_kWh | float64 | 0 |
| Lagging_Current_Reactive.Power_kVarh | float64 | 0 |
| Leading_Current_Reactive_Power_kVarh | float64 | 0 |
| CO2(tCO2) | float64 | 0 |
| Lagging_Current_Power_Factor | float64 | 0 |
| Leading_Current_Power_Factor | float64 | 0 |
| NSM | int64 | 0 |
| WeekStatus | object | 0 |
| Day_of_week | object | 0 |
| Load_Type | object | 0 |

## Target summary

| statistic | Usage_kWh |
| --- | --- |
| count | 35040.0 |
| mean | 27.387 |
| std | 33.444 |
| min | 0.0 |
| 25% | 3.2 |
| 50% | 4.57 |
| 75% | 51.238 |
| max | 157.18 |

## First five records

| date | Usage_kWh | Lagging_Current_Reactive.Power_kVarh | Leading_Current_Reactive_Power_kVarh | CO2(tCO2) | Lagging_Current_Power_Factor | Leading_Current_Power_Factor | NSM | WeekStatus | Day_of_week | Load_Type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-01-01 00:00:00 | 3.42 | 3.46 | 0.0 | 0.0 | 70.3 | 100.0 | 0 | Weekday | Monday | Light_Load |
| 2018-01-01 00:15:00 | 3.17 | 2.95 | 0.0 | 0.0 | 73.21 | 100.0 | 900 | Weekday | Monday | Light_Load |
| 2018-01-01 00:30:00 | 4.0 | 4.46 | 0.0 | 0.0 | 66.77 | 100.0 | 1800 | Weekday | Monday | Light_Load |
| 2018-01-01 00:45:00 | 3.24 | 3.28 | 0.0 | 0.0 | 70.28 | 100.0 | 2700 | Weekday | Monday | Light_Load |
| 2018-01-01 01:00:00 | 3.31 | 3.56 | 0.0 | 0.0 | 68.09 | 100.0 | 3600 | Weekday | Monday | Light_Load |

## Last five records

| date | Usage_kWh | Lagging_Current_Reactive.Power_kVarh | Leading_Current_Reactive_Power_kVarh | CO2(tCO2) | Lagging_Current_Power_Factor | Leading_Current_Power_Factor | NSM | WeekStatus | Day_of_week | Load_Type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-12-31 22:45:00 | 3.82 | 4.54 | 0.0 | 0.0 | 64.38 | 100.0 | 81900 | Weekday | Monday | Light_Load |
| 2018-12-31 23:00:00 | 3.85 | 4.86 | 0.0 | 0.0 | 62.1 | 100.0 | 82800 | Weekday | Monday | Light_Load |
| 2018-12-31 23:15:00 | 3.74 | 3.74 | 0.0 | 0.0 | 70.71 | 100.0 | 83700 | Weekday | Monday | Light_Load |
| 2018-12-31 23:30:00 | 3.78 | 3.17 | 0.07 | 0.0 | 76.62 | 99.98 | 84600 | Weekday | Monday | Light_Load |
| 2018-12-31 23:45:00 | 3.78 | 3.06 | 0.11 | 0.0 | 77.72 | 99.96 | 85500 | Weekday | Monday | Light_Load |

## Limitations

- One site and one calendar year; seasonality, shifts, and future years may differ.
- Production volume, product mix, machine state, maintenance events, tariffs, and weather are not supplied as direct features.
- Sensor features and `Load_Type` must be available for the forecast interval. This is a conditional, one-step forecast, not an unconditional long-range forecast.
- Historical lag features assume each preceding 15-minute usage reading is available.
- The source's original task labels `Load_Type` as target; do not confuse that classification setup with the regression target used here.
- `CO2(tCO2)` is included in the source description and EDA but excluded from predictors because it has a near-perfect target correlation and unclear timing.
- `Load_Type` is used as a model input only under the assumption that its future value is known independently of the measured Usage_kWh.
