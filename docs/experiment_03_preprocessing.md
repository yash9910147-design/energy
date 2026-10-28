# Experiment 3: Preprocessing and Feature Engineering

Run `python -m steel_energy.preprocessing` from the repository root. The
default source is the supplied `Steel_industry_data 3.csv` at the repository
root; the source path and row counts are recorded in
`data/processed/preprocessing_summary.json`.

The script parses the source's day-first timestamps, removes exact duplicate
rows, rejects ambiguous duplicate timestamps, sorts chronologically, and
reindexes to a complete 15-minute grid. Missing intervals remain explicit
rows with missing measurements; the target is never fabricated. Numeric
imputation and categorical encoding are fitted inside each training-fold
pipeline, not on validation or test data.

The source CO2 field remains in the cleaned source dataset but is intentionally
not part of engineered model inputs because its timing is unclear and its
near-perfect relationship with Usage_kWh could leak the contemporaneous target.

Calendar variables and shifted lag/rolling statistics are added before a
strict 70%/15%/15% chronological split. The first 96 rows without full
historical context are excluded from model splits. The complete cleaned and
engineered CSVs retain their original rows/grid; split files contain only rows
with a target and complete lag history.

Generated files under `data/processed/`:

- `cleaned_dataset.csv`
- `engineered_features.csv`
- `train.csv`, `validation.csv`, and `test.csv`
- `preprocessing_summary.json`

The fitted transform is saved as a versioned
`artifacts/models/preprocessor_vN.joblib` during model training. Each estimator is a scikit-learn Pipeline containing its
imputer, scaler, one-hot encoder, and regressor.
