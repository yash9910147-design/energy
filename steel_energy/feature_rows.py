from datetime import datetime

import numpy as np
import pandas as pd

from steel_energy.config import (
    FEATURE_COLUMNS,
    FORECAST_NUMERIC_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    TARGET,
    TIMESTAMP,
)


def build_inference_row(
    timestamp: datetime,
    sensor_values: dict[str, object],
    usage_history: list[float],
) -> pd.DataFrame:
    """Build the same lag/calendar feature row used by offline training."""
    if len(usage_history) < 96:
        raise ValueError("At least 96 previous 15-minute Usage_kWh readings are required.")

    row: dict[str, object] = {TIMESTAMP: timestamp, TARGET: np.nan}
    for column in FORECAST_NUMERIC_FEATURES:
        if column == "NSM":
            row[column] = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
            continue
        if column not in sensor_values:
            raise ValueError(f"Missing sensor input: {column}")
        row[column] = float(sensor_values[column])

    day_number = timestamp.weekday()
    row["WeekStatus"] = "Weekend" if day_number >= 5 else "Weekday"
    row["Day_of_week"] = timestamp.strftime("%A")
    row["Load_Type"] = sensor_values.get("Load_Type")
    row["hour"] = timestamp.hour
    row["day"] = timestamp.day
    row["day_of_week_num"] = day_number
    row["day_of_year"] = timestamp.timetuple().tm_yday
    row["month"] = timestamp.month
    row["minute"] = timestamp.minute
    row["is_weekend"] = int(day_number >= 5)
    row["lag_1"] = usage_history[-1]
    row["lag_4"] = usage_history[-4]
    row["lag_96"] = usage_history[-96]
    row["rolling_mean_4"] = float(np.mean(usage_history[-4:]))
    row["rolling_std_4"] = float(np.std(usage_history[-4:], ddof=1))
    row["rolling_mean_96"] = float(np.mean(usage_history[-96:]))

    if any(row.get(column) is None for column in RAW_CATEGORICAL_FEATURES):
        raise ValueError("Load_Type is required for each future interval.")
    return pd.DataFrame([{column: row[column] for column in [TIMESTAMP, TARGET, *FEATURE_COLUMNS]}])
