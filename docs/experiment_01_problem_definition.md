# Experiment 1: Problem Definition

## Problem statement

A steel plant needs a forecast of the electricity it will consume in each
upcoming 15-minute interval. Better short-horizon estimates can support
production scheduling, peak management, and electricity procurement.

## Machine-learning task

- **Task:** supervised regression with a chronological, time-series evaluation.
- **Target:** `Usage_kWh`, the measured interval energy consumption.
- **Frequency:** 15 minutes.
- **Prediction horizon:** one interval ahead for the operational API; recursive
  multi-interval prediction is supported when future sensor/load inputs are
  supplied.
- **Important source-data distinction:** UCI's dataset record lists
  `Load_Type` as the original categorical target. This project intentionally
  reframes the same data as regression: `Usage_kWh` is the target and
  `Load_Type` is an input feature.

## Inputs

The source readings used as interval inputs are:

- Lagging and leading reactive power (`Lagging_Current_Reactive.Power_kVarh`,
  `Leading_Current_Reactive_Power_kVarh`)
- Lagging and leading current power factor
- `NSM`, weekday/weekend, day name, and `Load_Type`
- Derived calendar values: hour, day, numeric day of week, day of year, month,
  minute, and weekend indicator
- Historical target values: Usage_kWh at lags 1, 4, and 96; prior-four-reading
  mean and standard deviation; prior-96-reading mean

The lag and rolling features are shifted so the target for the row being
predicted is never used to construct that row's inputs.

The source `CO2(tCO2)` field is retained for data description and EDA but
excluded from model inputs: its near-perfect correlation with Usage_kWh and
unclear measurement timing make target leakage a material risk.

## Objectives and success measures

1. Establish a reproducible baseline against the persistence forecast
   (next interval equals the last observed Usage_kWh).
2. Train and chronologically compare Random Forest and XGBoost regressors.
3. Select a model using validation RMSE, then report a final held-out test
   comparison.
4. Report MAE, MSE, RMSE, MAPE, and R-squared. MAPE excludes zero-usage rows
   because percentage error is undefined at zero; the excluded count is
   reported alongside it.
5. Keep preprocessing leakage-safe and provide a validated local prediction API.

No numerical acceptance threshold is claimed in advance: a plant operator
should set an acceptable kWh/RMSE limit based on procurement and operational
costs. The maintenance plan uses explicit starter alert thresholds that should
be reviewed with stakeholders.

## Stakeholders

- Plant energy manager and procurement team: forecast quality and peak-cost
  decisions.
- Production planners: expected use by operating schedule/load type.
- Electrical and instrumentation engineers: sensor quality and input
  availability.
- ML/system maintainer: model validation, logging, retraining, and rollback.

## Constraints and assumptions

- The supplied `Steel_industry_data 3.csv` contains one site's 2018 records
  only (35,040 readings), not
  production volume, product mix, equipment status, maintenance, tariffs, or
  weather.
- Interval sensor values and load type must be known or forecast separately
  for the future interval. This is a conditional prediction, not a pure
  long-range forecast.
- Historical usage readings are assumed to become available at each interval.
- Split data chronologically; never randomly shuffle it.
- Model output is a planning aid, not a replacement for plant safety or
  engineering controls.

## Why machine learning

Consumption can vary nonlinearly with operating load, reactive power,
power-factor readings, recent consumption, and calendar conditions. A supervised
regressor can combine these relationships without assuming that one linear
equation describes all operating regimes. Chronological holdout periods and a
persistence baseline make the claimed forecasting benefit measurable.
