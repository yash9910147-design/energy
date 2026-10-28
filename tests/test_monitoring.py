import pandas as pd

from steel_energy.config import FEATURE_COLUMNS
from steel_energy.monitoring import detect_data_drift, monitor_performance


def test_drift_detector_flags_shifted_numeric_feature() -> None:
    reference = pd.DataFrame({column: [0.0] * 100 for column in FEATURE_COLUMNS})
    reference["hour"] = [i % 24 for i in range(100)]
    current = reference.copy()
    current["hour"] = [23.0] * 100

    report = detect_data_drift(reference, current)

    assert "hour" in report["drifted_features"]


def test_drift_detector_waits_for_minimum_current_sample() -> None:
    reference = pd.DataFrame({column: [0.0] * 100 for column in FEATURE_COLUMNS})
    current = reference.iloc[:2]

    report = detect_data_drift(reference, current)

    assert report["status"] == "insufficient_data"
    assert report["drifted_features"] == []


def test_performance_monitor_does_not_trigger_before_minimum_sample() -> None:
    timestamps = pd.date_range("2018-01-01", periods=10, freq="15min")
    predictions = pd.DataFrame(
        {"timestamp": timestamps, "prediction_Usage_kWh": [10.0] * len(timestamps)}
    )
    predictions = pd.concat(
        [
            predictions,
            predictions.iloc[[0]].assign(prediction_Usage_kWh=100.0),
        ],
        ignore_index=True,
    )
    actuals = pd.DataFrame({"date": timestamps, "Usage_kWh": [20.0] * len(timestamps)})

    report = monitor_performance(predictions, actuals, validation_rmse=1.0)

    assert report["rows_matched"] == 10
    assert report["metrics"]["MAE"] == 17.0
    assert report["evaluation_required"] is False
