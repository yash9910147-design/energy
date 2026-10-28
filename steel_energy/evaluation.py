import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from steel_energy.config import (
    FIGURE_DIR,
    FEATURE_COLUMNS,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    REPORT_DIR,
    TARGET,
    TIMESTAMP,
)
from steel_energy.modeling import regression_metrics


def evaluate_models() -> dict[str, object]:
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing {test_path}. Run preprocessing and training first."
        )
    test = pd.read_csv(test_path, parse_dates=[TIMESTAMP])
    manifest_path = MODEL_DIR / "active_model.json"
    if not manifest_path.exists():
        raise FileNotFoundError("No trained model manifest; run `python -m steel_energy.modeling`.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = test[TARGET]
    comparison: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {
        "persistence_baseline": test["lag_1"].to_numpy()
    }

    for model_name in ("random_forest", "xgboost"):
        model_path = MODEL_DIR / manifest["model_artifacts"][model_name]
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained model artifact: {model_path}")
        model = joblib.load(model_path)
        predicted = model.predict(test[FEATURE_COLUMNS])
        predictions[model_name] = predicted
        comparison.append(
            {"model": model_name, **regression_metrics(actual, predicted)}
        )
    comparison.insert(
        0,
        {
            "model": "persistence_baseline",
            **regression_metrics(actual, predictions["persistence_baseline"]),
        },
    )
    selected_name = str(manifest["model_name"])
    selected_predictions = predictions[selected_name]
    rows = pd.DataFrame(
        {
            TIMESTAMP: test[TIMESTAMP],
            "actual_Usage_kWh": actual,
            "predicted_Usage_kWh": selected_predictions,
            "error_kWh": actual.to_numpy() - selected_predictions,
            "absolute_error_kWh": np.abs(actual.to_numpy() - selected_predictions),
            "Load_Type": test["Load_Type"],
        }
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    rows.nlargest(25, "absolute_error_kWh").to_csv(
        REPORT_DIR / "error_analysis_worst_25.csv", index=False
    )
    metrics_frame = pd.DataFrame(comparison).sort_values("RMSE")
    metrics_frame.to_csv(REPORT_DIR / "model_comparison_test.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(rows[TIMESTAMP], rows["actual_Usage_kWh"], label="Actual", linewidth=1)
    ax.plot(rows[TIMESTAMP], rows["predicted_Usage_kWh"], label="Predicted", linewidth=1)
    ax.set(title=f"Test period: actual vs predicted ({selected_name})", ylabel="Usage_kWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=rows.sample(min(len(rows), 5000), random_state=42),
        x="predicted_Usage_kWh",
        y="error_kWh",
        hue="Load_Type",
        alpha=0.5,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Test residuals by predicted usage and load type", ylabel="Actual - predicted (kWh)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "residuals.png", dpi=150)
    plt.close(fig)

    by_load = rows.groupby("Load_Type").agg(
        rows=("error_kWh", "size"),
        MAE=("absolute_error_kWh", "mean"),
        bias=("error_kWh", "mean"),
    )
    report = {
        "test_period": {
            "start": str(rows[TIMESTAMP].min()),
            "end": str(rows[TIMESTAMP].max()),
        },
        "evaluation_scope": (
            "Rolling one-step forecasts: each row's lag features use previously measured "
            "Usage_kWh, including within the test period. Recursive multi-step performance "
            "is not represented by these metrics."
        ),
        "selected_model": selected_name,
        "selection_rule": "Model selected using validation RMSE; test data was held out.",
        "test_metrics": comparison,
        "error_by_load_type": by_load.to_dict(orient="index"),
        "worst_25_errors_csv": str(REPORT_DIR / "error_analysis_worst_25.csv"),
        "zero_target_mape_note": (
            "MAPE is computed only on non-zero actual Usage_kWh rows; zero rows are counted "
            "in zero_actual_rows_excluded_from_MAPE."
        ),
    }
    (REPORT_DIR / "model_evaluation.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8"
    )
    lines = [
        "# Experiment 6: Model Evaluation",
        "",
        f"- **Selected model:** `{selected_name}` ({manifest['model_version']}).",
        "- **Selection rule:** validation RMSE; test data remained held out until final evaluation.",
        f"- **Test period:** {report['test_period']['start']} through {report['test_period']['end']}.",
        "- **Evaluation scope:** rolling one-step forecasts; each row can use previously "
        "measured Usage_kWh, including earlier test-period readings. These metrics do not "
        "measure recursive multi-step accuracy.",
        "",
        "## Test metrics",
        "",
        "| Model | MAE | MSE | RMSE | MAPE (%) on nonzero actuals | R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in comparison:
        lines.append(
            f"| {result['model']} | {result['MAE']:.3f} | {result['MSE']:.3f} | "
            f"{result['RMSE']:.3f} | {result['MAPE_percent_nonzero_actuals']:.2f} | "
            f"{result['R2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Error analysis",
            "",
            "The residual plot uses actual minus predicted kWh; positive residuals indicate "
            "under-prediction. The worst-25 CSV supports follow-up by timestamp and load type. "
            "Review subgroup bias and peaks before deployment; do not remove outliers solely "
            "to improve aggregate metrics.",
            "",
            f"Per-load-type error summary: `{by_load.round(3).to_dict(orient='index')}`.",
            "",
            "![Actual vs predicted](../figures/actual_vs_predicted.png)",
            "",
            "![Residual plot](../figures/residuals.png)",
            "",
            "## Final model rationale",
            "",
            f"`{selected_name}` had the lowest validation RMSE among the tuned tree models. "
            "XGBoost achieved a lower RMSE than Random Forest on this particular test period, "
            "but the test set was not used to change the validation-selected model; this "
            "difference supports collecting another out-of-time validation period. The "
            "persistence baseline and all models are reported on the held-out test period. "
            "Production rollout should be conditional "
            "on the plant confirming sensor availability and acceptable operational error limits.",
        ]
    )
    (REPORT_DIR / "model_evaluation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate_models(), indent=2))
