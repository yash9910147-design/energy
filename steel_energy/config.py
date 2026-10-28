from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "Steel_industry_data 3.csv"
UCI_DOWNLOAD_PATH = ROOT_DIR / "data" / "raw" / "steel_industry_data.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "artifacts" / "models"

TARGET = "Usage_kWh"
TIMESTAMP = "date"
FREQUENCY = "15min"

RAW_NUMERIC_FEATURES = [
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
]
RAW_CATEGORICAL_FEATURES = ["WeekStatus", "Day_of_week", "Load_Type"]
FORECAST_NUMERIC_FEATURES = [
    column for column in RAW_NUMERIC_FEATURES if column != "CO2(tCO2)"
]
CALENDAR_NUMERIC_FEATURES = [
    "hour",
    "day",
    "day_of_week_num",
    "day_of_year",
    "month",
    "minute",
    "is_weekend",
]
LAG_FEATURES = [
    "lag_1",
    "lag_4",
    "lag_96",
    "rolling_mean_4",
    "rolling_std_4",
    "rolling_mean_96",
]
NUMERIC_FEATURES = FORECAST_NUMERIC_FEATURES + CALENDAR_NUMERIC_FEATURES + LAG_FEATURES
CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/851/"
    "steel+industry+energy+consumption.zip"
)
UCI_DATASET_PAGE = (
    "https://archive.ics.uci.edu/dataset/851/"
    "steel+industry+energy+consumption"
)
