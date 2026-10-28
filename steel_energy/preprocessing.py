import json
import logging
from pathlib import Path

import pandas as pd

from steel_energy.config import (
    FEATURE_COLUMNS,
    FREQUENCY,
    LAG_FEATURES,
    PROCESSED_DATA_DIR,
    RAW_CATEGORICAL_FEATURES,
    RAW_DATA_PATH,
    RAW_NUMERIC_FEATURES,
    REPORT_DIR,
    TARGET,
    TIMESTAMP,
    UCI_DATASET_PAGE,
)

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = [TIMESTAMP, TARGET, *RAW_NUMERIC_FEATURES, *RAW_CATEGORICAL_FEATURES]
UCI_SOURCE = (
    "UCI Machine Learning Repository, Steel Industry Energy Consumption, "
    "dataset 851, DOI: 10.24432/C52G8C."
)
UCI_CREATORS = "Sathishkumar V E, Changsun Shin, and Yongyun Cho."
UCI_PUBLICATION = (
    "Efficient energy consumption prediction model for a data analytic-enabled "
    "industry building in a smart city, Building Research & Information, "
    "49(1), 127-143 (2021)."
)


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the original UCI CSV and parse its day-first timestamp."""
    data = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    parsed = pd.to_datetime(data[TIMESTAMP], format="%d/%m/%Y %H:%M", errors="coerce")
    invalid_timestamps = int(parsed.isna().sum())
    if invalid_timestamps:
        raise ValueError(f"Found {invalid_timestamps} invalid timestamps in {path}")
    data[TIMESTAMP] = parsed
    return data[REQUIRED_COLUMNS].copy()


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Sort, de-duplicate, and reindex records without inventing target readings."""
    cleaned = data.copy()
    initial_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates().copy()
    exact_duplicates_removed = initial_rows - len(cleaned)

    if cleaned[TIMESTAMP].duplicated().any():
        duplicate_count = int(cleaned[TIMESTAMP].duplicated(keep=False).sum())
        raise ValueError(
            f"Found {duplicate_count} records with duplicate timestamps; "
            "resolve conflicting measurements before preprocessing."
        )

    cleaned = cleaned.sort_values(TIMESTAMP).set_index(TIMESTAMP)
    expected_index = pd.date_range(
        start=cleaned.index.min(),
        end=cleaned.index.max(),
        freq=FREQUENCY,
        name=TIMESTAMP,
    )
    off_grid = cleaned.index.difference(expected_index)
    if len(off_grid):
        raise ValueError(
            f"Found {len(off_grid)} timestamps that are not aligned to {FREQUENCY}."
        )

    missing_intervals = len(expected_index.difference(cleaned.index))
    cleaned = cleaned.reindex(expected_index).reset_index()
    for column in RAW_NUMERIC_FEATURES + [TARGET]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    return cleaned, {
        "input_rows": initial_rows,
        "exact_duplicates_removed": exact_duplicates_removed,
        "missing_intervals_inserted": missing_intervals,
        "rows_with_missing_target": int(cleaned[TARGET].isna().sum()),
        "rows_with_missing_features": int(
            cleaned[RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES].isna().any(axis=1).sum()
        ),
    }


def engineer_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and strictly historical target features."""
    frame = cleaned.sort_values(TIMESTAMP).copy()
    timestamp = pd.to_datetime(frame[TIMESTAMP])
    frame["hour"] = timestamp.dt.hour
    frame["day"] = timestamp.dt.day
    frame["day_of_week_num"] = timestamp.dt.dayofweek
    frame["day_of_year"] = timestamp.dt.dayofyear
    frame["month"] = timestamp.dt.month
    frame["minute"] = timestamp.dt.minute
    frame["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype("int8")

    previous_usage = frame[TARGET].shift(1)
    frame["lag_1"] = frame[TARGET].shift(1)
    frame["lag_4"] = frame[TARGET].shift(4)
    frame["lag_96"] = frame[TARGET].shift(96)
    frame["rolling_mean_4"] = previous_usage.rolling(4, min_periods=4).mean()
    frame["rolling_std_4"] = previous_usage.rolling(4, min_periods=4).std()
    frame["rolling_mean_96"] = previous_usage.rolling(96, min_periods=96).mean()
    return frame[[TIMESTAMP, TARGET, *FEATURE_COLUMNS]]


def chronological_split(
    features: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split in timestamp order; no future row can enter an earlier split."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one.")

    usable = features.dropna(subset=[TARGET, *LAG_FEATURES]).copy()
    usable = usable.sort_values(TIMESTAMP).reset_index(drop=True)
    n_rows = len(usable)
    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))
    if min(train_end, validation_end - train_end, n_rows - validation_end) < 1:
        raise ValueError("Not enough usable records for three chronological data splits.")

    return (
        usable.iloc[:train_end].copy(),
        usable.iloc[train_end:validation_end].copy(),
        usable.iloc[validation_end:].copy(),
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_dataset_report(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    source_path: Path = RAW_DATA_PATH,
    report_dir: Path = REPORT_DIR,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    chronological = raw.sort_values(TIMESTAMP)
    target_summary = raw[TARGET].describe().round(3)
    dtypes = raw.dtypes.astype(str).rename("dtype").to_frame()
    missing = raw.isna().sum().rename("missing_values").to_frame()
    lines = [
        "# Experiment 2: Dataset Description",
        "",
        f"- **Local source file used:** `{source_path.name}`.",
        f"- **Source:** {UCI_SOURCE}",
        f"- **Creators:** {UCI_CREATORS}",
        f"- **Associated publication:** {UCI_PUBLICATION}",
        f"- **Dataset page:** {UCI_DATASET_PAGE}",
        "- **License:** The dataset 851 API metadata does not report a license. "
        "Verify the current UCI record's terms before redistribution; cite the dataset DOI.",
        f"- **Original shape:** {raw.shape[0]:,} rows x {raw.shape[1]} columns.",
        f"- **Cleaned time-indexed shape:** {cleaned.shape[0]:,} rows x {cleaned.shape[1]} columns.",
        "- **Coverage:** 15-minute readings for calendar year 2018.",
        "- **Regression target in this project:** `Usage_kWh` (continuous kWh). "
        "UCI's original published target is categorical `Load_Type`; this project "
        "reframes the data for energy regression and uses `Load_Type` as an input.",
        "",
        "## Feature dictionary",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        "| `date` | Local interval timestamp, recorded every 15 minutes. |",
        "| `Usage_kWh` | Industry electricity consumption for the interval (project target). |",
        "| `Lagging_Current_Reactive.Power_kVarh` | Lagging reactive energy measurement. |",
        "| `Leading_Current_Reactive_Power_kVarh` | Leading reactive energy measurement. |",
        "| `CO2(tCO2)` | CO2 field as named in the source dataset; preserve source units/definition. |",
        "| `Lagging_Current_Power_Factor` | Lagging current power factor (percent in source metadata). |",
        "| `Leading_Current_Power_Factor` | Leading current power factor (percent in source metadata). |",
        "| `NSM` | Number of seconds from midnight. |",
        "| `WeekStatus` | `Weekday` or `Weekend`. |",
        "| `Day_of_week` | Day name, Sunday through Saturday. |",
        "| `Load_Type` | Source load category: light, medium, or maximum load. |",
        "",
        "## Data types and missing values",
        "",
        _markdown_table(pd.concat([dtypes, missing], axis=1).reset_index(names="column")),
        "",
        "## Target summary",
        "",
        _markdown_table(target_summary.rename_axis("statistic").reset_index()),
        "",
        "## First five records",
        "",
        _markdown_table(chronological.head().round(3)),
        "",
        "## Last five records",
        "",
        _markdown_table(chronological.tail().round(3)),
        "",
        "## Limitations",
        "",
        "- One site and one calendar year; seasonality, shifts, and future years may differ.",
        "- Production volume, product mix, machine state, maintenance events, tariffs, "
        "and weather are not supplied as direct features.",
        "- Sensor features and `Load_Type` must be available for the forecast interval. "
        "This is a conditional, one-step forecast, not an unconditional long-range forecast.",
        "- Historical lag features assume each preceding 15-minute usage reading is available.",
        "- The source's original task labels `Load_Type` as target; do not confuse that "
        "classification setup with the regression target used here.",
        "- `CO2(tCO2)` is included in the source description and EDA but excluded from "
        "predictors because it has a near-perfect target correlation and unclear timing.",
        "- `Load_Type` is used as a model input only under the assumption that its future "
        "value is known independently of the measured Usage_kWh.",
    ]
    path = report_dir / "dataset_description.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_preprocessing(
    source_path: Path = RAW_DATA_PATH,
    output_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORT_DIR,
) -> dict[str, object]:
    raw = load_raw_dataset(source_path)
    cleaned, quality = clean_dataset(raw)
    engineered = engineer_features(cleaned)
    train, validation, test = chronological_split(engineered)

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_dir / "cleaned_dataset.csv", index=False)
    engineered.to_csv(output_dir / "engineered_features.csv", index=False)
    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(output_dir / "validation.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    report_path = write_dataset_report(raw, cleaned, source_path, report_dir)
    summary = {
        "source_file": str(source_path.resolve()),
        **quality,
        "cleaned_rows": len(cleaned),
        "engineered_rows": len(engineered),
        "usable_rows": len(train) + len(validation) + len(test),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "target": TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "first_timestamp": str(cleaned[TIMESTAMP].min()),
        "last_timestamp": str(cleaned[TIMESTAMP].max()),
        "dataset_report": str(report_path),
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Prepared %s intervals (%s usable after lag warm-up).",
        len(cleaned),
        summary["usable_rows"],
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(run_preprocessing(), indent=2))
