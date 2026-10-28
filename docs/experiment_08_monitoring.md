# Experiment 8: Model Monitoring and Maintenance

## Routine

- **Every prediction:** append timestamp, model version, prediction, and
  engineered input features to `outputs/logs/predictions.jsonl`.
- **Monthly:** join those predictions to measured `date,Usage_kWh` data;
  inspect MAE, RMSE, MAPE, R-squared, load-type bias, and worst errors.
- **Monthly and after process/equipment changes:** compare recent input
  distributions with the training reference using population stability index
  (PSI) once at least 100 recent predictions are available. Review each feature
  with PSI above 0.20; smaller batches are reported as insufficient data.
- Do not add the CO2 field to deployed model inputs without proving it is
  available before the prediction interval; its strong relationship to
  Usage_kWh is a potential leakage path.
- **After maintenance, new equipment, product-mix/production-volume changes,
  sensor recalibration, or sustained drift:** perform an out-of-time model
  evaluation and retraining review.

## Starter performance triggers

The monitoring script requests an evaluation when at least 100 matched
readings are available and either:

- RMSE exceeds 1.25 times validation RMSE; or
- MAPE exceeds 20% on non-zero actual readings.

These are initial alert thresholds, not guarantees of business suitability.
The plant energy owner must approve or revise them against acceptable kWh and
procurement risk before operational use. Zero actuals are included in MAE,
MSE, RMSE, and R-squared but excluded from MAPE.

## Commands

```bash
python -m steel_energy.monitoring \
  --predictions outputs/logs/predictions.jsonl \
  --actuals path/to/measured_actuals.csv
```

The actuals CSV must contain `date` and `Usage_kWh`. Drift reporting runs
whenever logged features are present; performance reporting runs when actuals
are supplied. Results are saved to
`outputs/reports/monitoring_report.json`.

The project also includes
`outputs/reports/monitoring_holdout_replay.json`, produced by running the
versioned model against all 5,242 held-out records and comparing those
predictions to the actual readings in the supplied dataset. This retrospective
check is separate from ongoing API prediction logs; a fresh production log
with fewer than 100 records correctly reports insufficient data for a drift
or performance alert.

## Retraining, versioning, and rollback

Run preprocessing and training again after an evaluation review. Training
creates new compressed `random_forest_vN.joblib` and `xgboost_vN.joblib`
artifacts plus a version-specific manifest, and updates
`active_model.json` only after a successful validation/tuning run. Keep prior
versioned artifacts and the matching report/logs. Roll back by restoring the
previous model version as the active model artifact and restoring its manifest,
then restart the API process. Verify `/health` and a representative prediction
before routing production traffic to a new version.

The ML maintainer owns monthly reports and artifact integrity; the plant
energy owner reviews error thresholds, operating-regime changes, and approval
to retrain or roll back. The sensor/controls team investigates input quality
and equipment changes.
