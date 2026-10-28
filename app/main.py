import json
import hmac
import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import unquote

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from steel_energy.config import (
    FEATURE_COLUMNS,
    LOG_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    REPORT_DIR,
    ROOT_DIR,
    UPLOAD_DIR,
    TARGET,
    TIMESTAMP,
)
from steel_energy.deployment_store import (
    download_artifact,
    load_active_version,
    publish_version,
)
from steel_energy.feature_rows import build_inference_row
from steel_energy.evaluation import evaluate_models
from steel_energy.eda import run_eda
from steel_energy.modeling import MODEL_GRIDS, regression_metrics, train_models
from steel_energy.preprocessing import REQUIRED_COLUMNS, run_preprocessing

LOGGER = logging.getLogger(__name__)
LOG_LOCK = Lock()
TRAIN_LOCK = Lock()
IS_VERCEL = os.environ.get("VERCEL") == "1"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 if IS_VERCEL else 25 * 1024 * 1024
LOAD_TYPES = Literal["Light_Load", "Medium_Load", "Maximum_Load"]
WEB_UI_PATH = Path(__file__).parent / "static" / "index.html"
_SYNCED_CLOUD_VERSION: str | None = None
PREDICTION_LOG = Path(
    os.environ.get(
        "STEEL_PREDICTION_LOG",
        str(Path("/tmp/steel_energy_predictions.jsonl") if IS_VERCEL else LOG_DIR / "predictions.jsonl"),
    )
)


def _seed_vercel_runtime() -> None:
    if not IS_VERCEL:
        return
    active_manifest = ROOT_DIR / "artifacts" / "models" / "active_model.json"
    if not active_manifest.is_file():
        raise FileNotFoundError(f"Bundled active model manifest is missing: {active_manifest}")
    manifest = json.loads(active_manifest.read_text(encoding="utf-8"))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (active_manifest, MODEL_DIR / active_manifest.name),
        (
            ROOT_DIR / "artifacts" / "models" / manifest["model_path"],
            MODEL_DIR / manifest["model_path"],
        ),
        (
            ROOT_DIR / "data" / "processed" / "test.csv",
            PROCESSED_DATA_DIR / "test.csv",
        ),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"Bundled runtime artifact is missing: {source}")
        shutil.copy2(source, destination)
    summary = ROOT_DIR / "outputs" / "reports" / f"upload_training_{manifest['model_version']}.json"
    if summary.is_file():
        shutil.copy2(summary, REPORT_DIR / summary.name)


_seed_vercel_runtime()

app = FastAPI(
    title="Steel Industry Energy Consumption Forecast API",
    description=(
        "Predict Usage_kWh for future 15-minute intervals. Supply the latest "
        "96 measured readings and the future interval sensor/load inputs."
    ),
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    if not WEB_UI_PATH.is_file():
        raise HTTPException(status_code=500, detail="The prediction web interface is missing.")
    return FileResponse(WEB_UI_PATH)


def _sync_vercel_training_state() -> None:
    global _SYNCED_CLOUD_VERSION
    if not IS_VERCEL or not os.environ.get("MONGODB_URI"):
        return
    try:
        state = load_active_version()
        if state is None:
            return
        version = str(state["model_version"])
        if version == _SYNCED_CLOUD_VERSION:
            return
        for pathname in state["artifacts"]:
            artifact_path = Path(str(pathname))
            if artifact_path.parts[0] == "models":
                destination = MODEL_DIR / artifact_path.name
            elif artifact_path.parts[0] == "processed":
                destination = PROCESSED_DATA_DIR / artifact_path.name
            elif artifact_path.parts[0] == "reports":
                destination = REPORT_DIR / artifact_path.name
            else:
                raise ValueError(f"Unsupported persisted model artifact: {pathname}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination.with_suffix(destination.suffix + ".tmp")
            temporary_path.write_bytes(download_artifact(str(pathname)))
            temporary_path.replace(destination)
        _SYNCED_CLOUD_VERSION = version
        _load_model.cache_clear()
    except Exception as error:
        LOGGER.exception("Could not synchronize trained artifacts from MongoDB.")
        raise HTTPException(
            status_code=503,
            detail=f"Could not load persisted model state from MongoDB: {error}",
        ) from error


def _require_training_authorization(request: Request) -> None:
    if not IS_VERCEL:
        return
    expected_token = os.environ.get("STEEL_TRAINING_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Set the STEEL_TRAINING_TOKEN environment variable to enable protected model training.",
        )
    authorization = request.headers.get("authorization", "")
    scheme, _, supplied_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied_token, expected_token):
        raise HTTPException(status_code=401, detail="A valid model-training access key is required.")


class HistoryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    usage_kwh: float = Field(ge=0, allow_inf_nan=False)


class FutureInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    timestamp: datetime
    lagging_current_reactive_power_kvarh: float = Field(
        alias="Lagging_Current_Reactive.Power_kVarh", ge=0, allow_inf_nan=False
    )
    leading_current_reactive_power_kvarh: float = Field(
        alias="Leading_Current_Reactive_Power_kVarh", ge=0, allow_inf_nan=False
    )
    lagging_current_power_factor: float = Field(
        alias="Lagging_Current_Power_Factor", ge=0, le=100, allow_inf_nan=False
    )
    leading_current_power_factor: float = Field(
        alias="Leading_Current_Power_Factor", ge=0, le=100, allow_inf_nan=False
    )
    load_type: LOAD_TYPES = Field(alias="Load_Type")

    @field_validator("timestamp")
    @classmethod
    def validate_quarter_hour_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError("Use timezone-naive local timestamps, matching the source data.")
        if value.minute not in (0, 15, 30, 45) or value.second or value.microsecond:
            raise ValueError("timestamp must align to a 15-minute interval.")
        return value

    def sensor_values(self) -> dict[str, object]:
        values = self.model_dump(by_alias=True, exclude={"timestamp"})
        return values


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: list[HistoryPoint] = Field(min_length=96, max_length=96)
    future_intervals: list[FutureInterval] = Field(min_length=1, max_length=96)

    @model_validator(mode="after")
    def validate_chronological_intervals(self) -> "PredictionRequest":
        history_timestamps = [point.timestamp for point in self.history]
        future_timestamps = [point.timestamp for point in self.future_intervals]
        for timestamp in history_timestamps:
            if timestamp.tzinfo is not None and timestamp.utcoffset() is not None:
                raise ValueError("Use timezone-naive local timestamps.")
            if timestamp.minute not in (0, 15, 30, 45) or timestamp.second or timestamp.microsecond:
                raise ValueError("All timestamps must align to a 15-minute interval.")
        all_timestamps = history_timestamps + future_timestamps
        if any(
            right - left != timedelta(minutes=15)
            for left, right in zip(all_timestamps, all_timestamps[1:])
        ):
            raise ValueError("History and future interval timestamps must be contiguous and ordered.")
        return self


class HistoricalTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def validate_quarter_hour_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError("Use timezone-naive local timestamps, matching the source data.")
        if value.minute not in (0, 15, 30, 45) or value.second or value.microsecond:
            raise ValueError("timestamp must align to a 15-minute interval.")
        return value


@lru_cache(maxsize=1)
def _load_model(model_version: str, relative_path: str):
    model_path = MODEL_DIR / relative_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact for {model_version} not found at {model_path}."
        )
    return joblib.load(model_path)


def _json_safe_features(frame: pd.DataFrame) -> dict[str, object]:
    values = frame[FEATURE_COLUMNS].iloc[0].to_dict()
    return {
        name: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
        for name, value in values.items()
    }


def _append_prediction_log(
    timestamp: datetime,
    prediction: float,
    model_version: str,
    feature_values: dict[str, object],
) -> None:
    record = {
        "timestamp": timestamp.isoformat(),
        "prediction_Usage_kWh": prediction,
        "model_version": model_version,
        "features": feature_values,
    }
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK, PREDICTION_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, allow_nan=False) + "\n")


@app.get("/health")
def health() -> dict[str, object]:
    _sync_vercel_training_state()
    manifest_path = MODEL_DIR / "active_model.json"
    model_ready = False
    model_version = None
    model_name = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_ready = (MODEL_DIR / manifest["model_path"]).is_file()
        if model_ready:
            model_version = manifest["model_version"]
            model_name = manifest["model_name"]
    return {
        "status": "ok",
        "model_ready": model_ready,
        "model_version": model_version,
        "model_name": model_name,
        "training_requires_auth": IS_VERCEL,
        "persistent_storage_ready": not IS_VERCEL or bool(os.environ.get("MONGODB_URI")),
    }


@app.post("/train")
async def train_uploaded_dataset(request: Request) -> dict[str, object]:
    global _SYNCED_CLOUD_VERSION
    _require_training_authorization(request)
    if IS_VERCEL and not os.environ.get("MONGODB_URI"):
        raise HTTPException(
            status_code=503,
            detail="MongoDB persistent storage is not configured for model training.",
        )
    _sync_vercel_training_state()
    if not TRAIN_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A dataset is already training. Wait for it to finish, then try again.",
        )

    try:
        content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip()
        if content_type not in {"text/csv", "application/csv", "application/octet-stream"}:
            raise HTTPException(
                status_code=415,
                detail="Upload a CSV file using the browser's dataset upload control.",
            )

        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"CSV upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB "
                        "project limit."
                    ),
                )
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

        filename = Path(unquote(request.headers.get("x-filename", "steel_dataset.csv"))).name
        if not filename.lower().endswith(".csv"):
            raise HTTPException(status_code=415, detail="The uploaded file must have a .csv extension.")

        current_manifest_path = MODEL_DIR / "active_model.json"
        previous_manifest = (
            json.loads(current_manifest_path.read_text(encoding="utf-8"))
            if current_manifest_path.is_file()
            else {"model_version": "v0"}
        )
        next_version = f"v{int(str(previous_manifest['model_version']).removeprefix('v')) + 1}"

        with tempfile.TemporaryDirectory(prefix="steel-energy-training-") as directory:
            workspace = Path(directory)
            uploaded_path = workspace / filename
            uploaded_path.write_bytes(payload)
            staged_data = workspace / "processed"
            staged_models = workspace / "models"
            staged_reports = workspace / "reports"

            try:
                preprocessing = await run_in_threadpool(
                    run_preprocessing,
                    uploaded_path,
                    staged_data,
                    staged_reports,
                )
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not preprocess the uploaded steel CSV: {error}",
                ) from error

            if preprocessing["usable_rows"] < 100:
                raise HTTPException(
                    status_code=422,
                    detail="The CSV needs at least 196 valid 15-minute rows for lag features and chronological model splits.",
                )

            try:
                training = await run_in_threadpool(
                    train_models,
                    split_dir=staged_data,
                    model_dir=staged_models,
                    report_dir=staged_reports,
                    version=next_version,
                    activate=False,
                )
            except (ValueError, OSError) as error:
                raise HTTPException(
                    status_code=422,
                    detail=f"Training could not complete for this dataset: {error}",
                ) from error

            test = pd.read_csv(staged_data / "test.csv", parse_dates=[TIMESTAMP])
            test_metrics: dict[str, dict[str, float | int]] = {
                "persistence_baseline": regression_metrics(
                    test[TARGET],
                    test["lag_1"].to_numpy(),
                )
            }
            for model_name in MODEL_GRIDS:
                model_path = staged_models / training["selection"]["model_artifacts"][model_name]
                candidate = joblib.load(model_path)
                test_metrics[model_name] = regression_metrics(
                    test[TARGET],
                    candidate.predict(test[FEATURE_COLUMNS]),
                )

            selected_name = str(training["selection"]["model_name"])
            selected_test_metrics = test_metrics[selected_name]
            training_report_path = staged_reports / "training_report.json"
            training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
            training_report["artifacts"] = {
                name: str(MODEL_DIR / path.name)
                for name, path in (
                    (model_name, staged_models / filename)
                    for model_name, filename in training["selection"]["model_artifacts"].items()
                )
            }
            training_report["best_model_path"] = str(
                MODEL_DIR / training["selection"]["model_path"]
            )
            training_report["preprocessor_path"] = str(
                MODEL_DIR / training["selection"]["preprocessor_path"]
            )
            training_report_path.write_text(
                json.dumps(training_report, indent=2, default=float) + "\n",
                encoding="utf-8",
            )
            upload_dir = UPLOAD_DIR
            upload_dir.mkdir(parents=True, exist_ok=True)
            persisted_dataset = upload_dir / f"{next_version}_{filename}"
            persisted_tmp = persisted_dataset.with_suffix(".csv.tmp")
            persisted_tmp.write_bytes(payload)
            persisted_tmp.replace(persisted_dataset)

            preprocessing["source_file"] = str(persisted_dataset.resolve())
            preprocessing["dataset_report"] = str(
                (REPORT_DIR / "dataset_description.md").resolve()
            )
            (staged_data / "preprocessing_summary.json").write_text(
                json.dumps(preprocessing, indent=2) + "\n",
                encoding="utf-8",
            )

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            for model_path in staged_models.glob("*.joblib"):
                shutil.copy2(model_path, MODEL_DIR / model_path.name)
            shutil.copy2(
                staged_models / f"model_manifest_{next_version}.json",
                MODEL_DIR / f"model_manifest_{next_version}.json",
            )
            for filename_to_copy in (
                "cleaned_dataset.csv",
                "engineered_features.csv",
                "preprocessing_summary.json",
                "train.csv",
                "validation.csv",
                "test.csv",
            ):
                shutil.copy2(staged_data / filename_to_copy, PROCESSED_DATA_DIR / filename_to_copy)
            for filename_to_copy in (
                "dataset_description.md",
                "model_comparison_validation.csv",
                "training_report.json",
                "validation_metrics.json",
            ):
                shutil.copy2(staged_reports / filename_to_copy, REPORT_DIR / filename_to_copy)

            active_manifest_tmp = MODEL_DIR / "active_model.json.tmp"
            active_manifest_tmp.write_text(
                json.dumps(training["selection"], indent=2) + "\n",
                encoding="utf-8",
            )
            active_manifest_tmp.replace(current_manifest_path)
            _load_model.cache_clear()

            await run_in_threadpool(run_eda, persisted_dataset)
            evaluation = await run_in_threadpool(evaluate_models)
            upload_report = {
                "source_file": str(persisted_dataset.resolve()),
                "model_version": next_version,
                "selected_model": selected_name,
                "selection_metric": "Lowest validation RMSE; test data was not used for model selection.",
                "validation_rmse": training["selection"]["validation_rmse"],
                "validation_metrics": {
                    name: record["validation_metrics"]
                    for name, record in training["models"].items()
                },
                "test_metrics": test_metrics,
                "test_comparison": evaluation["test_metrics"],
                "split_rows": preprocessing,
            }
            (REPORT_DIR / f"upload_training_{next_version}.json").write_text(
                json.dumps(upload_report, indent=2, default=float) + "\n",
                encoding="utf-8",
            )

            if IS_VERCEL:
                cloud_artifacts = {
                    f"models/{model_path.name}": model_path.read_bytes()
                    for model_path in MODEL_DIR.glob(f"*_{next_version}.joblib")
                }
                cloud_artifacts.update(
                    {
                        "models/active_model.json": current_manifest_path.read_bytes(),
                        "processed/test.csv": (PROCESSED_DATA_DIR / "test.csv").read_bytes(),
                        f"reports/upload_training_{next_version}.json": (
                            REPORT_DIR / f"upload_training_{next_version}.json"
                        ).read_bytes(),
                        "reports/training_report.json": (
                            REPORT_DIR / "training_report.json"
                        ).read_bytes(),
                    }
                )
                try:
                    publish_version(next_version, cloud_artifacts)
                    _SYNCED_CLOUD_VERSION = next_version
                except Exception as error:
                    LOGGER.exception("Could not persist trained artifacts to MongoDB.")
                    raise HTTPException(
                        status_code=503,
                        detail=f"Training finished but MongoDB could not save the new model: {error}",
                    ) from error

            if PREDICTION_LOG.is_file() and PREDICTION_LOG.stat().st_size:
                archived_log = PREDICTION_LOG.with_name(
                    f"predictions_before_{next_version}.jsonl"
                )
                PREDICTION_LOG.replace(archived_log)
            PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
            PREDICTION_LOG.touch()

            return {
                "status": "trained",
                "source_filename": filename,
                "source_path": str(persisted_dataset.resolve()),
                "input_rows": preprocessing["input_rows"],
                "usable_rows": preprocessing["usable_rows"],
                "split_rows": {
                    "train": preprocessing["train_rows"],
                    "validation": preprocessing["validation_rows"],
                    "test": preprocessing["test_rows"],
                },
                "model_version": next_version,
                "selected_model": selected_name,
                "selection_metric": "Lowest validation RMSE; held-out test data was not used for selection.",
                "validation_rmse": training["selection"]["validation_rmse"],
                "validation_metrics": {
                    name: record["validation_metrics"]
                    for name, record in training["models"].items()
                },
                "test_metrics": test_metrics,
                "selected_test_metrics": selected_test_metrics,
                "accuracy_note": (
                    "Regression has no classification accuracy score. MAPE is the percentage error "
                    "on non-zero actual test readings; lower is better."
                ),
            }
    finally:
        TRAIN_LOCK.release()


@app.get("/historical-test-info")
def historical_test_info() -> dict[str, object]:
    _sync_vercel_training_state()
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Held-out test data is missing at {test_path}. Run preprocessing first.",
        )
    test = pd.read_csv(test_path, usecols=[TIMESTAMP], parse_dates=[TIMESTAMP])
    return {
        "start_timestamp": test[TIMESTAMP].min().isoformat(),
        "end_timestamp": test[TIMESTAMP].max().isoformat(),
        "available_rows": len(test),
        "interval": "15 minutes",
        "description": (
            "Unseen held-out test period; this is historical evaluation, "
            "not a future live forecast."
        ),
    }


@app.get("/training-summary")
def training_summary() -> dict[str, object]:
    _sync_vercel_training_state()
    manifest_path = MODEL_DIR / "active_model.json"
    if not manifest_path.is_file():
        return {"available": False}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = REPORT_DIR / f"upload_training_{manifest['model_version']}.json"
    if not report_path.is_file():
        return {"available": False}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    preprocessing = report["split_rows"]
    return {
        "available": True,
        "source_filename": Path(report["source_file"]).name,
        "input_rows": preprocessing["input_rows"],
        "split_rows": {
            "train": preprocessing["train_rows"],
            "validation": preprocessing["validation_rows"],
            "test": preprocessing["test_rows"],
        },
        "model_version": report["model_version"],
        "selected_model": report["selected_model"],
        "validation_rmse": report["validation_rmse"],
        "validation_metrics": report["validation_metrics"],
        "test_metrics": report["test_metrics"],
        "selected_test_metrics": report["test_metrics"][report["selected_model"]],
    }


@app.post("/historical-test")
def historical_test(request: HistoricalTestRequest) -> dict[str, object]:
    _sync_vercel_training_state()
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Held-out test data is missing at {test_path}. Run preprocessing first.",
        )

    test = pd.read_csv(test_path, parse_dates=[TIMESTAMP])
    match = test.loc[test[TIMESTAMP] == request.timestamp]
    if match.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "Choose a 15-minute timestamp within the held-out test period "
                f"({test[TIMESTAMP].min().isoformat()} to {test[TIMESTAMP].max().isoformat()})."
            ),
        )

    manifest_path = MODEL_DIR / "active_model.json"
    if not manifest_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="No active model is available. Train a model first.",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        model = _load_model(
            str(manifest["model_version"]),
            str(manifest["model_path"]),
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    row = match.iloc[[0]]
    actual = float(row[TARGET].iloc[0])
    predicted = max(0.0, float(model.predict(row[FEATURE_COLUMNS])[0]))
    return {
        "timestamp": request.timestamp.isoformat(),
        "actual_Usage_kWh": actual,
        "predicted_Usage_kWh": predicted,
        "absolute_error_kWh": abs(actual - predicted),
        "model_version": manifest["model_version"],
        "evaluation": (
            "One-step prediction on a held-out historical test record "
            "using earlier measured history."
        ),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    _sync_vercel_training_state()
    manifest_path = MODEL_DIR / "active_model.json"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Active model manifest not found at {manifest_path}. Train a model first.",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        model = _load_model(
            str(manifest["model_version"]),
            str(manifest["model_path"]),
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    usage_history = [point.usage_kwh for point in request.history]
    predictions: list[dict[str, object]] = []
    for interval in request.future_intervals:
        feature_row = build_inference_row(
            interval.timestamp,
            interval.sensor_values(),
            usage_history,
        )
        prediction = max(0.0, float(model.predict(feature_row[FEATURE_COLUMNS])[0]))
        features = _json_safe_features(feature_row)
        _append_prediction_log(
            interval.timestamp,
            prediction,
            str(manifest["model_version"]),
            features,
        )
        predictions.append(
            {"timestamp": interval.timestamp.isoformat(), "predicted_Usage_kWh": prediction}
        )
        usage_history.append(prediction)

    LOGGER.info(
        "Generated %s prediction(s) with model %s.",
        len(predictions),
        manifest["model_version"],
    )
    return {
        "model_version": manifest["model_version"],
        "target": "Usage_kWh",
        "frequency": "15 minutes",
        "predictions": predictions,
    }
