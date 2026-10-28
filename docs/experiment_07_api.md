# Experiment 7: Local Prediction API

Start the API from the repository root after training:

```bash
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The `POST
/predict` request accepts exactly 96 consecutive previous measurements and
one or more consecutive future 15-minute intervals (up to 96). Each future interval requires four source sensor measurements and `Load_Type`. Calendar
fields and `NSM` are derived from the timestamp.

The [web interface](http://127.0.0.1:8000/) has two distinct workflows:

- **Upload and train:** selecting a training CSV calls `POST /train`
  automatically. The complete steel schema is required: `date`, `Usage_kWh`,
  both reactive-power fields, both power-factor fields, `CO2(tCO2)`, `NSM`,
  `WeekStatus`, `Day_of_week`, and `Load_Type`. The endpoint creates
  chronological train/validation/test splits, tunes Random Forest and XGBoost
  with `TimeSeriesSplit`, selects by validation RMSE, evaluates on the held-out
  test set, saves a versioned model, and activates it after successful
  training. The response includes MAE, RMSE, MAPE, R-squared, and model
  comparisons.
- **Future forecast:** after successful training, the latest 96 actual
  `date`/`Usage_kWh` readings from the uploaded file are used for lag features.
  Supply the next interval's sensor readings and load type to get a forecast.
  Future sensor values are never silently invented.
- **Historical test:** choose a timestamp from the held-out test period to
  compare the prediction with that record's actual `Usage_kWh`. This is a
  one-step historical evaluation, not a future live forecast.

Training expects the full 11-column steel dataset, sorted internally by time.
The forecast history must contain at least 96 unique, consecutive 15-minute
readings with non-negative `Usage_kWh`. Future power-factor and reactive-power
readings must be available or forecast separately; the energy model does not
predict those inputs. Uploads are limited to 25 MB.

The upload body is the CSV bytes (`Content-Type: text/csv`); `X-Filename` is
an optional display/storage name. Training uses a chronological validation
score to choose a model, never the held-out test score. The test comparison
may therefore show a different model with lower error without changing the
selected model.

Example request builder (uses the source data for real history and sample
future sensor inputs):

```python
import json
import pandas as pd
from urllib.request import Request, urlopen

data = pd.read_csv("data/raw/steel_industry_data.csv", encoding="utf-8-sig")
data["timestamp"] = pd.to_datetime(data["date"], dayfirst=True)
history_rows = data.sort_values("timestamp").tail(96)
history = [
    {
        "timestamp": row.timestamp.isoformat(),
        "usage_kwh": float(row.Usage_kWh),
    }
    for row in history_rows.itertuples(index=False)
]
last_sample = history_rows.iloc[-1]
next_timestamp = last_sample["timestamp"] + pd.Timedelta(minutes=15)
payload = {
    "history": history,
    "future_intervals": [{
        "timestamp": next_timestamp.isoformat(),
        "Lagging_Current_Reactive.Power_kVarh": float(last_sample["Lagging_Current_Reactive.Power_kVarh"]),
        "Leading_Current_Reactive_Power_kVarh": float(last_sample["Leading_Current_Reactive_Power_kVarh"]),
        "Lagging_Current_Power_Factor": float(last_sample["Lagging_Current_Power_Factor"]),
        "Leading_Current_Power_Factor": float(last_sample["Leading_Current_Power_Factor"]),
        "Load_Type": last_sample["Load_Type"],
    }],
}
request = Request(
    "http://127.0.0.1:8000/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=30) as response:
    print(response.status, json.loads(response.read()))
```

For multi-interval output, add up to 95 more future items with consecutive
15-minute timestamps and all required sensor/load fields. Later steps use
earlier predictions as lag inputs.

The API always uses user-supplied history and future sensor/load inputs for
live forecasts. Historical evaluation is separately restricted to timestamps
in the held-out test split and returns both actual and predicted energy.

Example response from the trained local model:

```json
{
  "model_version": "v3",
  "target": "Usage_kWh",
  "frequency": "15 minutes",
  "predictions": [
    {
      "timestamp": "2019-01-01T00:00:00",
      "predicted_Usage_kWh": 3.691963214990985
    }
  ]
}
```

The API response is the observed result from model version `v3`; rerun the
request after training to obtain the result for your current model version.
The application returns HTTP 503 with an actionable
message until a trained model artifact exists. Invalid timestamps, load
categories, ranges, extra fields, and non-contiguous intervals are rejected.
Successful calls are appended to `outputs/logs/predictions.jsonl`.
