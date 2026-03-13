from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


TENURE_BUCKET_BINS = [-1, 12, 24, 48, float("inf")]
TENURE_BUCKET_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
MONTHLY_CHARGE_BUCKET_BINS = [-0.01, 35, 70, 90, float("inf")]
MONTHLY_CHARGE_BUCKET_LABELS = ["0-35", "36-70", "71-90", "90+"]


def assign_tenure_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.astype(float),
        bins=TENURE_BUCKET_BINS,
        labels=TENURE_BUCKET_LABELS,
    )


def assign_monthly_charge_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.astype(float),
        bins=MONTHLY_CHARGE_BUCKET_BINS,
        labels=MONTHLY_CHARGE_BUCKET_LABELS,
    )


def segment_probability(probability: float) -> str:
    if probability < 0.35:
        return "Low Risk"
    if probability < 0.65:
        return "Medium Risk"
    return "High Risk"


def calculate_probability_percentile(
    probability: float,
    reference_scores: Iterable[float],
) -> float:
    scores = np.asarray(list(reference_scores), dtype=float)
    if scores.size == 0:
        return 0.0
    return float((scores <= probability).mean() * 100)


def classify_error(actual_value: int, predicted_value: int) -> str:
    if actual_value == 1 and predicted_value == 1:
        return "True Positive"
    if actual_value == 1 and predicted_value == 0:
        return "False Negative"
    if actual_value == 0 and predicted_value == 1:
        return "False Positive"
    return "True Negative"


def enrich_scored_frame(
    raw_df: pd.DataFrame,
    probabilities: pd.Series | np.ndarray | list[float],
    threshold: float,
    actual: pd.Series | np.ndarray | list[int] | None = None,
) -> pd.DataFrame:
    frame = raw_df.reset_index(drop=True).copy()
    probability_series = pd.Series(probabilities, dtype=float).reset_index(drop=True)

    frame["predicted_probability"] = probability_series
    frame["predicted_label"] = (probability_series >= threshold).astype(int)
    frame["predicted_label_name"] = frame["predicted_label"].map({0: "Stay", 1: "Churn"})
    frame["risk_segment"] = probability_series.apply(segment_probability)

    if "tenure" in frame.columns:
        frame["tenure_bucket"] = assign_tenure_bucket(frame["tenure"])
    if "MonthlyCharges" in frame.columns:
        monthly_charges = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").fillna(0.0)
        frame["monthly_charge_bucket"] = assign_monthly_charge_bucket(monthly_charges)
        frame["expected_monthly_revenue_at_risk"] = monthly_charges * probability_series
        frame["expected_annual_revenue_at_risk"] = frame["expected_monthly_revenue_at_risk"] * 12

    if actual is not None:
        actual_series = pd.Series(actual, dtype=int).reset_index(drop=True)
        frame["actual_label"] = actual_series
        frame["actual_label_name"] = frame["actual_label"].map({0: "Stay", 1: "Churn"})
        frame["error_type"] = [
            classify_error(actual_value, predicted_value)
            for actual_value, predicted_value in zip(actual_series, frame["predicted_label"])
        ]

    return frame


def summarize_segments(
    scored_df: pd.DataFrame,
    dimensions: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for dimension in dimensions:
        if dimension not in scored_df.columns:
            continue

        grouped = scored_df.groupby(dimension, dropna=False, observed=False)
        for segment, group in grouped:
            row: dict[str, float | int | str] = {
                "dimension": dimension,
                "segment": "Missing" if pd.isna(segment) else str(segment),
                "customer_count": int(len(group)),
                "avg_predicted_probability": float(group["predicted_probability"].mean()),
                "high_risk_rate": float((group["risk_segment"] == "High Risk").mean()),
            }

            if "MonthlyCharges" in group.columns:
                row["monthly_revenue"] = float(group["MonthlyCharges"].sum())
                row["expected_monthly_revenue_at_risk"] = float(
                    group["expected_monthly_revenue_at_risk"].sum()
                )
                row["avg_monthly_charges"] = float(group["MonthlyCharges"].mean())

            if "actual_label" in group.columns:
                row["actual_churn_rate"] = float(group["actual_label"].mean())
                row["actual_churn_count"] = int(group["actual_label"].sum())

            if "error_type" in group.columns:
                actual_positive_count = int((group["actual_label"] == 1).sum())
                actual_negative_count = int((group["actual_label"] == 0).sum())
                false_negative_count = int((group["error_type"] == "False Negative").sum())
                false_positive_count = int((group["error_type"] == "False Positive").sum())
                row["false_negative_count"] = false_negative_count
                row["false_positive_count"] = false_positive_count
                row["false_negative_rate"] = (
                    false_negative_count / actual_positive_count if actual_positive_count else 0.0
                )
                row["false_positive_rate"] = (
                    false_positive_count / actual_negative_count if actual_negative_count else 0.0
                )

            rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    sort_columns = [
        column
        for column in [
            "expected_monthly_revenue_at_risk",
            "avg_predicted_probability",
            "customer_count",
        ]
        if column in summary.columns
    ]
    return summary.sort_values(sort_columns, ascending=False).reset_index(drop=True)
