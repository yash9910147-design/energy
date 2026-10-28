import pandas as pd
import pytest

from steel_energy.config import (
    FEATURE_COLUMNS,
    FORECAST_NUMERIC_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    TARGET,
    TIMESTAMP,
)
from steel_energy.feature_rows import build_inference_row
from steel_energy.preprocessing import (
    clean_dataset,
    chronological_split,
    engineer_features,
)


def _sample_frame(rows: int = 120) -> pd.DataFrame:
    timestamps = pd.date_range("2018-01-01", periods=rows, freq="15min")
    values: dict[str, object] = {
        TIMESTAMP: timestamps,
        TARGET: [float(i + 1) for i in range(rows)],
        "WeekStatus": ["Weekday"] * rows,
        "Day_of_week": ["Monday"] * rows,
        "Load_Type": ["Light_Load"] * rows,
    }
    for column in RAW_NUMERIC_FEATURES:
        values[column] = [1.0] * rows
    return pd.DataFrame(values)


def test_clean_dataset_inserts_missing_intervals_without_filling_target() -> None:
    data = _sample_frame(10).drop(index=4)
    cleaned, quality = clean_dataset(data)

    assert len(cleaned) == 10
    assert pd.isna(cleaned.loc[4, TARGET])
    assert quality["missing_intervals_inserted"] == 1


def test_clean_dataset_removes_exact_duplicate_rows() -> None:
    data = _sample_frame(10)
    data = pd.concat([data, data.iloc[[3]]], ignore_index=True)

    cleaned, quality = clean_dataset(data)

    assert len(cleaned) == 10
    assert quality["exact_duplicates_removed"] == 1


def test_features_use_only_prior_usage_values() -> None:
    cleaned, _ = clean_dataset(_sample_frame(120))
    features = engineer_features(cleaned)

    assert features.loc[96, "lag_1"] == 96.0
    assert features.loc[96, "lag_4"] == 93.0
    assert features.loc[96, "lag_96"] == 1.0
    assert features.loc[96, "rolling_mean_4"] == 94.5
    assert pd.isna(features.loc[0, "lag_1"])


def test_chronological_split_is_ordered_and_disjoint() -> None:
    cleaned, _ = clean_dataset(_sample_frame(400))
    features = engineer_features(cleaned)
    train, validation, test = chronological_split(features)

    assert train[TIMESTAMP].max() < validation[TIMESTAMP].min()
    assert validation[TIMESTAMP].max() < test[TIMESTAMP].min()
    assert len(train) + len(validation) + len(test) == 400 - 96


def test_inference_features_match_offline_feature_names() -> None:
    timestamp = pd.Timestamp("2018-01-02 00:00")
    sensor_values = {
        **{column: 1.0 for column in FORECAST_NUMERIC_FEATURES if column != "NSM"},
        "Load_Type": "Light_Load",
    }
    row = build_inference_row(timestamp, sensor_values, list(range(1, 97)))

    assert row.loc[0, "lag_1"] == 96
    assert row.loc[0, "lag_4"] == 93
    assert row.loc[0, "lag_96"] == 1
    assert row.loc[0, "Day_of_week"] == "Tuesday"
    assert row.loc[0, "WeekStatus"] == "Weekday"
    assert row.loc[0, "NSM"] == 0
    assert len(row.columns) == 2 + len(FEATURE_COLUMNS)


def test_duplicate_timestamps_with_conflicting_readings_are_rejected() -> None:
    data = _sample_frame(10)
    conflicting = data.iloc[[3]].copy()
    conflicting[TARGET] = 999.0
    data = pd.concat([data, conflicting], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        clean_dataset(data)
