import calendar
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from steel_energy.config import FIGURE_DIR, RAW_DATA_PATH, REPORT_DIR, TARGET, TIMESTAMP
from steel_energy.preprocessing import load_raw_dataset


def run_eda(data_path: Path = RAW_DATA_PATH) -> dict[str, object]:
    data = load_raw_dataset(data_path).sort_values(TIMESTAMP).set_index(TIMESTAMP)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    daily = data[TARGET].resample("D").sum()
    hourly = data[TARGET].groupby(data.index.hour).mean()
    monthly_mean = data[TARGET].groupby(data.index.month).mean()
    weekday = data[TARGET].groupby(data.index.day_name()).mean()
    daily_load = data.groupby("Load_Type")[TARGET].agg(["count", "mean", "median"])
    numeric = data.select_dtypes(include="number")
    correlations = numeric.corr(numeric_only=True)[TARGET].drop(TARGET)
    lag_correlations = {
        str(lag): float(data[TARGET].autocorr(lag=lag))
        for lag in (1, 4, 96, 672)
    }
    skewness = float(data[TARGET].skew())
    q1, q3 = data[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_share = float(
        ((data[TARGET] < q1 - 1.5 * iqr) | (data[TARGET] > q3 + 1.5 * iqr)).mean()
    )
    weekend_means = data.groupby("WeekStatus")[TARGET].mean()
    strongest_feature = str(correlations.abs().idxmax())
    highest_month = int(monthly_mean.idxmax())
    lowest_month = int(monthly_mean.idxmin())

    fig, ax = plt.subplots(figsize=(13, 4))
    daily.plot(ax=ax, linewidth=0.8)
    ax.set(title="Daily steel-industry energy consumption", ylabel="Usage_kWh per day")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "consumption_timeseries.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    hourly.plot(ax=axes[0], marker="o")
    axes[0].set(title="Average energy consumption by hour", xlabel="Hour", ylabel="Mean Usage_kWh")
    weekday.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).plot(
        kind="bar", ax=axes[1]
    )
    axes[1].set(title="Average energy consumption by weekday", xlabel="", ylabel="Mean Usage_kWh")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "daily_weekly_patterns.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    monthly_mean.plot(kind="bar", ax=ax)
    ax.set(
        title="Average energy consumption by month",
        xlabel="Month",
        ylabel="Mean Usage_kWh per interval",
        xticklabels=[calendar.month_abbr[int(month)] for month in monthly_mean.index],
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "monthly_seasonality.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data[TARGET], bins=60, kde=True, ax=ax)
    ax.set(title="Usage_kWh distribution", xlabel="Usage_kWh")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=data, x="Load_Type", y=TARGET, ax=ax)
    ax.set(title="Energy consumption by load type", xlabel="Load type", ylabel="Usage_kWh")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "load_type_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(
        data=data.sample(min(len(data), 5000), random_state=42),
        x="Lagging_Current_Power_Factor",
        y=TARGET,
        hue="Load_Type",
        alpha=0.45,
        ax=axes[0],
        legend=False,
    )
    sns.scatterplot(
        data=data.sample(min(len(data), 5000), random_state=42),
        x="Leading_Current_Power_Factor",
        y=TARGET,
        hue="Load_Type",
        alpha=0.45,
        ax=axes[1],
        legend=False,
    )
    axes[0].set_title("Lagging power factor vs. energy")
    axes[1].set_title("Leading power factor vs. energy")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "power_factor_relationships.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric.corr(numeric_only=True), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric feature correlation")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    autocorrelation = pd.Series(
        {lag: data[TARGET].autocorr(lag=lag) for lag in range(1, 97)},
        name="autocorrelation",
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    autocorrelation.plot(ax=ax)
    ax.set(title="Usage_kWh autocorrelation (first 24 hours)", xlabel="15-minute lag", ylabel="Correlation")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "autocorrelation.png", dpi=150)
    plt.close(fig)

    insights = [
        f"Usage_kWh has skewness {skewness:.2f}; the distribution is "
        f"{'right-skewed' if skewness > 0 else 'left-skewed'} relative to a symmetric distribution.",
        f"Mean usage is {data[TARGET].mean():.2f} kWh per interval and median is "
        f"{data[TARGET].median():.2f} kWh, indicating that the median is a useful robust reference.",
        f"Weekend mean usage is {weekend_means.get('Weekend', float('nan')):.2f} kWh and "
        f"weekday mean usage is {weekend_means.get('Weekday', float('nan')):.2f} kWh.",
        f"{daily_load['mean'].idxmax()} has the highest average usage "
        f"({daily_load['mean'].max():.2f} kWh per interval) among the recorded load types.",
        f"Monthly average usage is highest in {calendar.month_name[highest_month]} "
        f"({monthly_mean.max():.2f} kWh) and lowest in {calendar.month_name[lowest_month]} "
        f"({monthly_mean.min():.2f} kWh); one year cannot establish a recurring annual season.",
        f"`{strongest_feature}` has the largest absolute Pearson correlation with Usage_kWh "
        f"among the numeric source features (r={correlations[strongest_feature]:.3f}).",
        f"Autocorrelation is {lag_correlations['1']:.3f} at 15 minutes and "
        f"{lag_correlations['96']:.3f} at 24 hours, supporting chronological lag features.",
        f"About {outlier_share:.1%} of target readings lie outside the 1.5-IQR whiskers; "
        "inspect these operating peaks rather than deleting them automatically.",
    ]
    report = {
        "target_summary": data[TARGET].describe().to_dict(),
        "skewness": skewness,
        "iqr_outlier_share": outlier_share,
        "weekday_means": weekend_means.to_dict(),
        "monthly_mean_usage": monthly_mean.to_dict(),
        "load_type_summary": daily_load.to_dict(orient="index"),
        "target_correlations": correlations.sort_values(key=abs, ascending=False).to_dict(),
        "autocorrelations": lag_correlations,
        "insights": insights,
    }
    (REPORT_DIR / "eda_summary.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8"
    )
    insight_text = "\n".join(f"{index}. {insight}" for index, insight in enumerate(insights, 1))
    (REPORT_DIR / "eda_insights.md").write_text(
        "# Experiment 4: EDA Insights\n\n"
        "## Findings\n\n"
        f"{insight_text}\n\n"
        "## Implications for modeling\n\n"
        "- Preserve chronological ordering and compare against a persistence baseline.\n"
        "- Include calendar and lag features; fit all imputers/encoders on training folds only.\n"
        "- Keep `Load_Type` and power-factor readings only when available at forecast time.\n"
        "- Exclude `CO2(tCO2)` from predictors until its timing is proven; its strong target correlation "
        "could reveal same-interval energy use.\n"
        "- Retain operational peaks and evaluate errors by load type and time period.\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = run_eda()
    print("\n".join(result["insights"]))
