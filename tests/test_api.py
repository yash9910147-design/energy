import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException
from starlette.requests import Request
from pydantic import ValidationError

from app.main import (
    FutureInterval,
    HistoricalTestRequest,
    PredictionRequest,
    historical_test,
    historical_test_info,
    home,
    predict,
    train_uploaded_dataset,
)
from steel_energy.config import RAW_DATA_PATH, TIMESTAMP, TARGET


def _future(timestamp: datetime) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "Lagging_Current_Reactive.Power_kVarh": 3.0,
        "Leading_Current_Reactive_Power_kVarh": 0.1,
        "Lagging_Current_Power_Factor": 75.0,
        "Leading_Current_Power_Factor": 99.0,
        "Load_Type": "Light_Load",
    }


def test_api_request_accepts_contiguous_history_and_future() -> None:
    start = datetime(2018, 1, 1)
    history = [
        {"timestamp": start + timedelta(minutes=15 * index), "usage_kwh": float(index)}
        for index in range(96)
    ]
    request = PredictionRequest(
        history=history,
        future_intervals=[_future(history[-1]["timestamp"] + timedelta(minutes=15))],
    )

    assert len(request.history) == 96
    assert request.future_intervals[0].load_type == "Light_Load"


def test_api_rejects_non_contiguous_future_timestamp() -> None:
    start = datetime(2018, 1, 1)
    history = [
        {"timestamp": start + timedelta(minutes=15 * index), "usage_kwh": float(index)}
        for index in range(96)
    ]

    with pytest.raises(ValidationError, match="contiguous"):
        PredictionRequest(
            history=history,
            future_intervals=[_future(history[-1]["timestamp"] + timedelta(minutes=30))],
        )


def test_api_rejects_invalid_power_factor() -> None:
    with pytest.raises(ValidationError):
        FutureInterval.model_validate(
            _future(datetime(2018, 1, 1, 0, 15))
            | {"Lagging_Current_Power_Factor": 101}
        )


def test_frontend_offers_real_history_upload_and_historical_test() -> None:
    page_response = home()
    page = Path(page_response.path)
    assert page.is_file()
    assert "Forecast plant energy use" in page.read_text()
    content = page.read_text()
    assert 'type="file"' in content
    assert 'fetch("/train"' in content
    assert 'fetch("/demo-input")' not in content
    assert "Historical test" in content


def test_historical_test_returns_prediction_and_actual_from_held_out_data() -> None:
    info = historical_test_info()
    test_data = pd.read_csv("data/processed/test.csv", parse_dates=[TIMESTAMP])
    expected = test_data.iloc[0]
    response = historical_test(
        HistoricalTestRequest(timestamp=expected[TIMESTAMP].to_pydatetime())
    )

    assert info["available_rows"] == len(test_data)
    assert response["actual_Usage_kWh"] == pytest.approx(float(expected[TARGET]))
    assert response["predicted_Usage_kWh"] >= 0
    assert response["absolute_error_kWh"] == pytest.approx(
        abs(response["actual_Usage_kWh"] - response["predicted_Usage_kWh"])
    )


def test_historical_test_rejects_timestamp_outside_held_out_period() -> None:
    with pytest.raises(HTTPException) as error:
        historical_test(HistoricalTestRequest(timestamp=datetime(2018, 1, 1)))

    assert error.value.status_code == 422


def test_live_prediction_accepts_user_supplied_history_and_sensor_values() -> None:
    data = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
    data[TIMESTAMP] = pd.to_datetime(data[TIMESTAMP], dayfirst=True)
    recent = data.sort_values(TIMESTAMP).tail(96)
    history = [
        {"timestamp": row[TIMESTAMP].to_pydatetime(), "usage_kwh": float(row[TARGET])}
        for _, row in recent.iterrows()
    ]
    latest = recent.iloc[-1]
    request = PredictionRequest(
        history=history,
        future_intervals=[
            _future(
                latest[TIMESTAMP].to_pydatetime() + timedelta(minutes=15)
            )
            | {
                "Lagging_Current_Reactive.Power_kVarh": float(
                    latest["Lagging_Current_Reactive.Power_kVarh"]
                ),
                "Leading_Current_Reactive_Power_kVarh": float(
                    latest["Leading_Current_Reactive_Power_kVarh"]
                ),
                "Lagging_Current_Power_Factor": float(
                    latest["Lagging_Current_Power_Factor"]
                ),
                "Leading_Current_Power_Factor": float(
                    latest["Leading_Current_Power_Factor"]
                ),
                "Load_Type": latest["Load_Type"],
            }
        ],
    )

    response = predict(request)

    assert len(response["predictions"]) == 1
    assert response["predictions"][0]["predicted_Usage_kWh"] >= 0


def test_configured_training_source_is_the_user_supplied_file() -> None:
    assert RAW_DATA_PATH.name == "Steel_industry_data 3.csv"
    assert RAW_DATA_PATH.is_file()


def _upload_request(csv_bytes: bytes, filename: str = "plant.csv") -> Request:
    headers = [
        (b"content-type", b"text/csv"),
        (b"x-filename", filename.encode()),
    ]
    delivered = False

    async def receive() -> dict[str, object]:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": csv_bytes, "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/train",
        "raw_path": b"/train",
        "query_string": b"",
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope, receive)


def test_training_upload_rejects_csv_without_required_training_columns() -> None:
    request = _upload_request(b"date,Usage_kWh\n01/01/2025 00:00,1.0\n")

    with pytest.raises(HTTPException) as error:
        asyncio.run(train_uploaded_dataset(request))

    assert error.value.status_code == 422
    assert "missing required columns" in error.value.detail.lower()
