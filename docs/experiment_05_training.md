# Experiment 5: Model Training

Run `python -m steel_energy.modeling` after preprocessing.

The training program compares Random Forest and XGBoost with
`GridSearchCV(TimeSeriesSplit(n_splits=3))`. Hyperparameter search uses only
the chronological training partition. The validation partition selects the
best model; all selected estimators are then refitted on train + validation
for final evaluation. The test period is not used for tuning or model
selection.

A persistence forecast (`lag_1`) is included as a simple operational
time-series baseline. Prophet and LSTM are not added: the short-horizon,
sensor-conditioned objective is directly evaluated with lagged regressors,
and those libraries would add a separate heavyweight runtime without
improving the required tree-model comparison.

Artifacts and reports:

- `artifacts/models/random_forest_vN.joblib` and
  `artifacts/models/xgboost_vN.joblib` (compressed, versioned artifacts)
- The selected model is the best-model checkpoint referenced by
  `artifacts/models/active_model.json`; a version-specific
  `model_manifest_vN.json` supports rollback
- `artifacts/models/preprocessor_vN.joblib`
- `outputs/reports/model_comparison_validation.csv`
- `outputs/reports/training_report.json` and `validation_metrics.json`
- `outputs/logs/training.log`

The chosen model is selected by validation RMSE, not the held-out test score.
