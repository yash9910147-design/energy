import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from steel_energy.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    LOG_DIR,
    MODEL_DIR,
    NUMERIC_FEATURES,
    PROCESSED_DATA_DIR,
    REPORT_DIR,
    TARGET,
    TIMESTAMP,
)

LOGGER = logging.getLogger(__name__)
MODEL_GRIDS = {
    "random_forest": (
        RandomForestRegressor(random_state=42, n_jobs=1),
        {
            "regressor__n_estimators": [100, 150],
            "regressor__max_leaf_nodes": [256, 512],
        },
    ),
    "xgboost": (
        XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=1,
            random_state=42,
            verbosity=0,
        ),
        {
            "regressor__n_estimators": [150],
            "regressor__max_depth": [3, 5],
            "regressor__learning_rate": [0.05],
        },
    ),
}


def build_pipeline(regressor: object) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float | int]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    nonzero = actual_values != 0
    mape = (
        float(mean_absolute_percentage_error(actual_values[nonzero], predicted_values[nonzero]) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {
        "MAE": float(mean_absolute_error(actual_values, predicted_values)),
        "MSE": float(mean_squared_error(actual_values, predicted_values)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "MAPE_percent_nonzero_actuals": mape,
        "R2": float(r2_score(actual_values, predicted_values)),
        "rows": int(len(actual_values)),
        "zero_actual_rows_excluded_from_MAPE": int((~nonzero).sum()),
    }


def _read_split(name: str, split_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    path = split_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m steel_energy.preprocessing` first."
        )
    return pd.read_csv(path, parse_dates=[TIMESTAMP])


def train_models(
    split_dir: Path = PROCESSED_DATA_DIR,
    model_dir: Path = MODEL_DIR,
    report_dir: Path = REPORT_DIR,
    version: str | None = None,
    activate: bool = True,
) -> dict[str, object]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "training.log", mode="a", encoding="utf-8"),
        ],
        force=True,
    )

    train = _read_split("train", split_dir)
    validation = _read_split("validation", split_dir)
    test = _read_split("test", split_dir)
    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    x_validation, y_validation = validation[FEATURE_COLUMNS], validation[TARGET]
    cv = TimeSeriesSplit(n_splits=3)
    validation_records: list[dict[str, object]] = []
    model_records: dict[str, dict[str, object]] = {}

    persistence = regression_metrics(y_validation, validation["lag_1"].to_numpy())
    validation_records.append({"model": "persistence_baseline", **persistence})

    for model_name, (regressor, parameter_grid) in MODEL_GRIDS.items():
        LOGGER.info("Tuning %s using three chronological CV folds.", model_name)
        search = GridSearchCV(
            estimator=build_pipeline(regressor),
            param_grid=parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        validation_predictions = search.best_estimator_.predict(x_validation)
        validation_result = regression_metrics(y_validation, validation_predictions)
        validation_records.append({"model": model_name, **validation_result})
        model_records[model_name] = {
            "best_params": search.best_params_,
            "best_cv_rmse": float(-search.best_score_),
            "validation_metrics": validation_result,
        }
        LOGGER.info(
            "%s: best CV RMSE %.3f; validation RMSE %.3f.",
            model_name,
            -search.best_score_,
            validation_result["RMSE"],
        )

    scores = pd.DataFrame(validation_records).sort_values("RMSE")
    validation_by_name = {
        str(row["model"]): float(row["RMSE"])
        for row in validation_records
    }
    selected_name = min(
        MODEL_GRIDS,
        key=lambda name: validation_by_name[name],
    )

    combined = pd.concat([train, validation], ignore_index=True)
    if version is None:
        previous_manifest_path = model_dir / "active_model.json"
        previous_version = (
            json.loads(previous_manifest_path.read_text(encoding="utf-8"))["model_version"]
            if previous_manifest_path.exists()
            else "v0"
        )
        version_number = int(str(previous_version).removeprefix("v")) + 1
        version = f"v{version_number}"
    saved_paths: dict[str, str] = {}
    for model_name, (regressor, _) in MODEL_GRIDS.items():
        best_params = model_records[model_name]["best_params"]
        configured_regressor = regressor.set_params(
            **{
                key.removeprefix("regressor__"): value
                for key, value in best_params.items()
            }
        )
        final_pipeline = build_pipeline(configured_regressor)
        final_pipeline.fit(combined[FEATURE_COLUMNS], combined[TARGET])
        model_path = model_dir / f"{model_name}_{version}.joblib"
        joblib.dump(final_pipeline, model_path, compress=3)
        saved_paths[model_name] = str(model_path)
        if model_name == selected_name:
            joblib.dump(
                final_pipeline.named_steps["preprocessor"],
                model_dir / f"preprocessor_{version}.joblib",
                compress=3,
            )
    best_path = Path(saved_paths[selected_name])
    manifest = {
        "model_version": version,
        "model_name": selected_name,
        "model_path": best_path.name,
        "model_artifacts": {
            model_name: Path(path).name for model_name, path in saved_paths.items()
        },
        "preprocessor_path": f"preprocessor_{version}.joblib",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "validation RMSE (lower is better)",
        "validation_rmse": validation_by_name[selected_name],
        "test_period_used_during_selection": False,
    }
    model_manifest_path = model_dir / f"model_manifest_{version}.json"
    model_manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if activate:
        active_manifest_tmp = model_dir / "active_model.json.tmp"
        active_manifest_tmp.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        active_manifest_tmp.replace(model_dir / "active_model.json")
    scores.to_csv(report_dir / "model_comparison_validation.csv", index=False)
    report = {
        "target": TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test_reserved_for_final_evaluation": len(test),
        },
        "selection": manifest,
        "models": model_records,
        "validation_metrics": validation_records,
        "artifacts": saved_paths,
        "best_model_path": str(best_path),
        "preprocessor_path": str(MODEL_DIR / manifest["preprocessor_path"]),
        "tuning": "GridSearchCV with TimeSeriesSplit(n_splits=3), training partition only.",
        "baseline": "Persistence forecast: next interval Usage_kWh equals lag_1.",
    }
    (report_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "validation_metrics.json").write_text(
        json.dumps(
            {
                "selected_model": selected_name,
                "selected_model_rmse": validation_by_name[selected_name],
                "models": {r["model"]: r for r in validation_records},
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Selected %s (%s) using validation RMSE.", selected_name, version)
    return report


if __name__ == "__main__":
    print(json.dumps(train_models(), indent=2))
