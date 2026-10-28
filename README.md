# Predicting Steel-Plant Energy Consumption

This folder contains a complete, reproducible ML lab project for predicting
electricity use at a steel manufacturing plant. It packages the steel-energy
code, supplied dataset, experiment outputs, trained model artifacts, and API;
it can be used independently of the original repository.

## Dataset and target

The steel workflow uses the real [UCI Steel Industry Energy Consumption
dataset](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption),
dataset 851, DOI [10.24432/C52G8C](https://doi.org/10.24432/C52G8C). Its
creators are Sathishkumar V E, Changsun Shin, and Yongyun Cho. The associated
publication is "Efficient energy consumption prediction model for a data
analytic-enabled industry building in a smart city" (Building Research &
Information, 49(1), 127-143, 2021). Its
35,040 records cover 15-minute intervals in 2018. The original source labels
`Load_Type` as a classification target. For this project's requested
regression task, **`Usage_kWh` is the target**, and `Load_Type` is an input
only when its future value is known independently of the interval's measured
energy. The highly correlated `CO2(tCO2)` field is retained for EDA but
excluded from model predictors because its timing could leak the target.
UCI's dataset API does not expose a license field; check the current source
record's terms before redistributing the data.

The project uses the supplied `Steel_industry_data 3.csv` at the repository
root as its training and evaluation source. The included UCI CSV at
`data/raw/steel_industry_data.csv` is byte-for-byte identical. To download a
fresh copy from UCI:

```bash
python -m steel_energy.download
```

## Setup on macOS / VS Code

Use Python 3.11 or 3.12. From the repository root, create and activate a
virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, choose `.venv/bin/python` as the workspace interpreter.

Run the focused project tests with:

```bash
python -m pytest -q tests
```

## Run all experiments

Run commands from the repository root:

```bash
# Experiments 1 and 2: read the problem definition and generated dataset report
open docs/experiment_01_problem_definition.md

# Experiment 3: clean, engineer, and chronologically split the real data
python -m steel_energy.preprocessing

# Experiment 4: open notebooks/steel_energy_eda.ipynb and run all cells
# or generate its graphs and insight reports directly:
python -m steel_energy.eda

# Experiment 5: time-series cross-validation, tuning, and model artifacts
python -m steel_energy.modeling

# Experiment 6: held-out test evaluation, graphs, and reports
python -m steel_energy.evaluation

# Experiment 7: start the local prediction API
uvicorn app.main:app --reload
# Frontend: http://127.0.0.1:8000/ (select a full steel CSV to auto-train and evaluate)
# Swagger UI: http://127.0.0.1:8000/docs

# Experiment 8: monitor prediction drift and, when available, actual errors
python -m steel_energy.monitoring \
  --predictions outputs/logs/predictions.jsonl \
  --actuals path/to/measured_actuals.csv
```

The preprocessing and training stages are independent commands; run them in
the order shown. Model training uses Random Forest and XGBoost with
`TimeSeriesSplit`; a persistence baseline is reported for comparison. A model
is selected using validation RMSE, then evaluated once on the held-out future
test period. All transformations are fit within the training pipeline.

Selecting a CSV in the web interface automatically runs preprocessing,
chronological splitting, model tuning, held-out test evaluation, and model
activation. It requires the complete steel dataset schema; a CSV containing
only timestamps and energy readings cannot retrain the sensor-conditioned
model.

## Eight experiment outputs

The complete all-in-one notebook—with code and executed outputs for each
experiment—is [notebooks/all_8_experiments.ipynb](notebooks/all_8_experiments.ipynb).
For a direct file-by-file map, open [EXPERIMENTS.md](EXPERIMENTS.md).

1. **Problem definition:** `docs/experiment_01_problem_definition.md`
2. **Dataset description:** `outputs/reports/dataset_description.md`
3. **Preprocessing:** `steel_energy/preprocessing.py`,
   `docs/experiment_03_preprocessing.md`, and `data/processed/`
4. **EDA:** `notebooks/steel_energy_eda.ipynb`,
   `outputs/figures/`, and `outputs/reports/eda_insights.md`
5. **Training:** `steel_energy/modeling.py`,
   `docs/experiment_05_training.md`, `artifacts/models/`, and training logs
6. **Evaluation:** `steel_energy/evaluation.py` and
   `outputs/reports/model_evaluation.md`
7. **Deployment:** `app/main.py` and `docs/experiment_07_api.md`
8. **Monitoring:** `steel_energy/monitoring.py` and
   `docs/experiment_08_monitoring.md`

Generated datasets are written to `data/processed/`, charts and reports to
`outputs/`, models to `artifacts/models/`, and prediction logs to
`outputs/logs/`. Generated files are ignored by Git; the original UCI CSV is
included locally under `data/raw/`.

## Operational assumptions and limitations

The model forecasts Usage_kWh conditional on the future interval's reactive
power, power-factor, and load-type inputs. Those readings must therefore be
known or forecast separately. Multi-step API forecasts recursively use
predicted energy for lag inputs. The source covers only one plant and one year
and does not include production volume, equipment mix, maintenance, tariffs,
or weather; assess these limits before real plant use.
