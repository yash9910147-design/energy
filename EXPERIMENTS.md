# Experiment Guide

This project contains all eight experiments. Their code and results are grouped
by type (for example, Python code is in `steel_energy/` and generated results
are in `outputs/`), rather than in eight separate folders. Use this guide to
find each experiment's files.

## One notebook with all experiments and outputs

- Open [all_8_experiments.ipynb](notebooks/all_8_experiments.ipynb) for the
  complete sequence, runnable code, real-data outputs, tables, graphs, saved
  model results, API example, and monitoring results in one notebook.

## Experiment 1 — Problem Definition

- Document: [Problem definition](docs/experiment_01_problem_definition.md)

## Experiment 2 — Dataset Description

- Dataset: [Steel industry CSV](Steel_industry_data%203.csv)
- Dataset report: [Dataset description](outputs/reports/dataset_description.md)
- Dataset report generator: [Preprocessing script](steel_energy/preprocessing.py)

## Experiment 3 — Preprocessing and Feature Engineering

- Code: [Preprocessing script](steel_energy/preprocessing.py)
- Guide: [Preprocessing notes](docs/experiment_03_preprocessing.md)
- Processed data and chronological splits: [Processed datasets](data/processed/)
- Pipeline and split summary: [Training report](outputs/reports/training_report.json)

## Experiment 4 — Exploratory Data Analysis

- Notebook: [EDA notebook](notebooks/steel_energy_eda.ipynb)
- Code: [EDA script](steel_energy/eda.py)
- Findings: [EDA insights](outputs/reports/eda_insights.md)
- Summary statistics: [EDA summary](outputs/reports/eda_summary.json)
- Charts: [Generated figures](outputs/figures/)

## Experiment 5 — Model Implementation and Training

- Code: [Model training script](steel_energy/modeling.py)
- Guide: [Training notes](docs/experiment_05_training.md)
- Training results and parameters: [Training report](outputs/reports/training_report.json)
- Model comparison: [Validation comparison](outputs/reports/model_comparison_validation.csv)
- Saved models and preprocessor: [Model artifacts](artifacts/models/)

## Experiment 6 — Model Evaluation

- Code: [Evaluation script](steel_energy/evaluation.py)
- Report: [Model evaluation](outputs/reports/model_evaluation.md)
- Metrics: [Evaluation results](outputs/reports/model_evaluation.json)
- Actual vs predicted and residual plots: [Evaluation figures](outputs/figures/)
- Detailed errors: [Worst test errors](outputs/reports/error_analysis_worst_25.csv)

## Experiment 7 — Model Deployment

- API and frontend: [FastAPI application](app/main.py) and [frontend files](app/static/)
- Guide and sample requests: [API documentation](docs/experiment_07_api.md)
- Run from this folder with `uvicorn app.main:app --reload`; open
  `http://127.0.0.1:8000/` for the frontend or
  `http://127.0.0.1:8000/docs` for Swagger UI.

## Experiment 8 — Monitoring and Maintenance

- Code: [Monitoring script](steel_energy/monitoring.py)
- Guide: [Monitoring and maintenance plan](docs/experiment_08_monitoring.md)
- Historical monitoring results: [Holdout replay](outputs/reports/monitoring_holdout_replay.json)
- Prediction and training logs: [Logs](outputs/logs/)

## Run the Experiment 4 notebook

From this project's folder, activate the project's Python environment and run:

```bash
jupyter notebook notebooks/steel_energy_eda.ipynb
```

The notebook already contains its executed analysis and charts. To regenerate
the EDA charts and summary reports from the included dataset instead, run:

```bash
python -m steel_energy.eda
```
