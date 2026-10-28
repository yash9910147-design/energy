import json
from pathlib import Path

import numpy as np
import pandas as pd

from steel_energy.config import FEATURE_COLUMNS, MODEL_DIR, REPORT_DIR, TARGET, TIMESTAMP
from steel_energy.modeling import regression_metrics

PREDICTION_TIMESTAMP = "timestamp"
DRIFT_THRESHOLD = 0.20
MINIMUM_DRIFT_ROWS = 100
MINIMUM_EVALUATION_ROWS = 100
RMSE_TRIGGER_MULTIPLIER = 1.25
MAPE_TRIGGER_PERCENT = 20.0


def _population_stability_index(reference: pd.Series, current: pd.Series) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return float("nan")
    quantiles = np.unique(reference.quantile(np.linspace(0, 1, 11)).to_numpy())
    inner_edges = quantiles[1:-1]
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)
    reference_share = (reference_counts + 1e-6) / (reference_counts.sum() + 1e-6 * len(edges[:-1]))
    current_share = (current_counts + 1e-6) / (current_counts.sum() + 1e-6 * len(edges[:-1]))
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = DRIFT_THRESHOLD,
) -> dict[str, object]:
    """Calculate PSI per feature and flag values above the supplied threshold."""
    if len(current) < MINIMUM_DRIFT_ROWS:
        return {
            "threshold": threshold,
            "minimum_current_rows": MINIMUM_DRIFT_ROWS,
            "reference_rows": int(len(reference)),
            "current_rows": int(len(current)),
            "status": "insufficient_data",
            "features": {},
            "drifted_features": [],
        }

    feature_results: dict[str, dict[str, object]] = {}
    for column in FEATURE_COLUMNS:
        if column not in reference or column not in current:
            continue
        if pd.api.types.is_numeric_dtype(reference[column]):
            score = _population_stability_index(reference[column], current[column])
        else:
            ref_share = reference[column].fillna("__missing__").astype(str).value_counts(normalize=True)
            cur_share = current[column].fillna("__missing__").astype(str).value_counts(normalize=True)
            categories = ref_share.index.union(cur_share.index)
            left = ref_share.reindex(categories, fill_value=0).to_numpy() + 1e-6
            right = cur_share.reindex(categories, fill_value=0).to_numpy() + 1e-6
            left /= left.sum()
            right /= right.sum()
            score = float(np.sum((right - left) * np.log(right / left)))
        feature_results[column] = {
            "psi": score,
            "drift_detected": bool(np.isfinite(score) and score > threshold),
        }
    return {
        "threshold": threshold,
        "minimum_current_rows": MINIMUM_DRIFT_ROWS,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "status": "evaluated",
        "features": feature_results,
        "drifted_features": [
            name for name, result in feature_results.items() if result["drift_detected"]
        ],
    }


def read_prediction_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction log not found: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return pd.DataFrame(
            columns=[PREDICTION_TIMESTAMP, "prediction_Usage_kWh", "features"]
        )
    return pd.DataFrame(records)


def monitor_performance(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    validation_rmse: float,
) -> dict[str, object]:
    if TIMESTAMP not in actuals or TARGET not in actuals:
        raise ValueError(f"Actual data must contain `{TIMESTAMP}` and `{TARGET}` columns.")
    if PREDICTION_TIMESTAMP not in predictions or "prediction_Usage_kWh" not in predictions:
        raise ValueError("Prediction records must contain timestamp and prediction_Usage_kWh.")
    actual_frame = actuals[[TIMESTAMP, TARGET]].copy()
    prediction_frame = predictions[
        [PREDICTION_TIMESTAMP, "prediction_Usage_kWh"]
    ].copy()
    actual_frame[TIMESTAMP] = pd.to_datetime(actual_frame[TIMESTAMP], dayfirst=True)
    prediction_frame[PREDICTION_TIMESTAMP] = pd.to_datetime(
        prediction_frame[PREDICTION_TIMESTAMP]
    )
    prediction_frame = prediction_frame.drop_duplicates(
        PREDICTION_TIMESTAMP, keep="last"
    )
    actual_frame = actual_frame.rename(columns={TIMESTAMP: PREDICTION_TIMESTAMP})
    joined = prediction_frame.merge(
        actual_frame,
        on=PREDICTION_TIMESTAMP,
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("No prediction timestamps match the supplied actual measurements.")
    metrics = regression_metrics(joined[TARGET], joined["prediction_Usage_kWh"].to_numpy())
    enough_rows = len(joined) >= MINIMUM_EVALUATION_ROWS
    rmse_threshold = validation_rmse * RMSE_TRIGGER_MULTIPLIER
    triggers = []
    if enough_rows and metrics["RMSE"] > rmse_threshold:
        triggers.append("RMSE exceeds 1.25 times validation RMSE")
    if (
        enough_rows
        and np.isfinite(metrics["MAPE_percent_nonzero_actuals"])
        and metrics["MAPE_percent_nonzero_actuals"] > MAPE_TRIGGER_PERCENT
    ):
        triggers.append("MAPE exceeds 20% on non-zero actuals")
    return {
        "rows_matched": int(len(joined)),
        "metrics": metrics,
        "minimum_rows_for_trigger": MINIMUM_EVALUATION_ROWS,
        "rmse_threshold": rmse_threshold,
        "mape_threshold_percent": MAPE_TRIGGER_PERCENT,
        "evaluation_required": bool(triggers),
        "triggers": triggers,
    }


def run_monitoring(prediction_log: Path, actuals_path: Path | None = None) -> dict[str, object]:
    predictions = read_prediction_log(prediction_log)
    logged_count = len(predictions)
    if not predictions.empty:
        predictions = predictions.drop_duplicates(PREDICTION_TIMESTAMP, keep="last")
    results: dict[str, object] = {
        "prediction_log_records": logged_count,
        "unique_prediction_timestamps": len(predictions),
    }

    feature_records = []
    for record in predictions.to_dict(orient="records"):
        features = record.get("features")
        if isinstance(features, dict):
            feature_records.append(features)
    if feature_records:
        reference_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "train.csv"
        if not reference_path.exists():
            raise FileNotFoundError(f"Training reference data not found: {reference_path}")
        reference = pd.read_csv(reference_path)
        drift = detect_data_drift(reference, pd.DataFrame(feature_records))
        results["drift"] = drift
        results["drift_evaluation_required"] = bool(drift["drifted_features"])

    if actuals_path is not None:
        manifest_path = MODEL_DIR / "active_model.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
        validation_path = REPORT_DIR / "validation_metrics.json"
        if not validation_path.exists():
            raise FileNotFoundError(f"Validation metrics not found: {validation_path}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        actuals = pd.read_csv(actuals_path)
        results["performance"] = monitor_performance(
            predictions,
            actuals,
            float(validation["selected_model_rmse"]),
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "monitoring_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check recent predictions for drift and error.")
    parser.add_argument("--predictions", type=Path, help="JSONL prediction log")
    parser.add_argument("--actuals", type=Path, help="CSV with date and Usage_kWh columns")
    args = parser.parse_args()
    from steel_energy.config import LOG_DIR

    results = run_monitoring(
        args.predictions or LOG_DIR / "predictions.jsonl",
        args.actuals,
    )
    print(json.dumps(results, indent=2))
