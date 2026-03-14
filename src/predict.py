from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import (
    calculate_probability_percentile,
    enrich_scored_frame,
    segment_probability,
)

from src.preprocessing import (
    transform_features,
    validate_input_frame,
)

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

def load_model_artifact(model_path: Path | str = MODEL_PATH) -> dict[str, Any]:

    artifact = joblib.load(model_path)

    if (
        not isinstance(artifact, dict)
        or "model" not in artifact
        or "preprocessing" not in artifact
    ):
        raise ValueError(
            "Invalid model artifact. Run `python src/train_model.py` to rebuild."
        )

    return artifact


# ---------------------------------------------------------
# INPUT PREPARATION
# ---------------------------------------------------------

def prepare_input_frame(input_data: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([input_data])


# ---------------------------------------------------------
# FEATURE DISPLAY NAMES
# ---------------------------------------------------------

def format_feature_name(feature_name: str) -> str:

    friendly_names = {
        "SeniorCitizen": "Senior Citizen",
        "tenure": "Tenure",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "PaperlessBilling": "Paperless Billing",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "PaymentMethod": "Payment Method",
    }

    if feature_name in friendly_names:
        return friendly_names[feature_name]

    if "_" in feature_name:
        prefix, suffix = feature_name.split("_", 1)
        prefix = friendly_names.get(prefix, prefix)
        return f"{prefix}: {suffix}"

    return feature_name


# ---------------------------------------------------------
# ARTIFACT RESOLUTION
# ---------------------------------------------------------

def _resolve_artifact(
    artifact: dict[str, Any] | None,
    model_path: Path | str,
) -> dict[str, Any]:

    return artifact if artifact is not None else load_model_artifact(model_path)


# ---------------------------------------------------------
# SCORING PIPELINE
# ---------------------------------------------------------

def score_dataframe(
    input_df: pd.DataFrame,
    model_path: Path | str = MODEL_PATH,
    artifact: dict[str, Any] | None = None,
    strict: bool = False,
) -> pd.DataFrame:

    resolved_artifact = _resolve_artifact(artifact, model_path)

    preprocessing = resolved_artifact["preprocessing"]
    model = resolved_artifact["model"]
    threshold = float(resolved_artifact.get("decision_threshold", 0.5))

    validated_input = validate_input_frame(input_df, preprocessing, strict=strict)

    transformed_input = transform_features(
        validated_input,
        preprocessing,
        strict=True,
    )

    probabilities = pd.Series(
        model.predict_proba(transformed_input)[:, 1],
        dtype=float,
    )

    scored = enrich_scored_frame(
        validated_input,
        probabilities,
        threshold=threshold,
    )

    reference_scores = resolved_artifact.get("reference_customer_scores", [])

    scored["probability_percentile"] = scored["predicted_probability"].apply(
        lambda p: calculate_probability_percentile(p, reference_scores)
    )

    scored["model_name"] = resolved_artifact["model_name"]
    scored["decision_threshold"] = threshold

    return scored


# ---------------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------------

def predict_batch(
    input_df: pd.DataFrame,
    model_path: Path | str = MODEL_PATH,
    artifact: dict[str, Any] | None = None,
    strict: bool = False,
) -> pd.DataFrame:

    return score_dataframe(
        input_df=input_df,
        model_path=model_path,
        artifact=artifact,
        strict=strict,
    )


# ---------------------------------------------------------
# SINGLE CUSTOMER PREDICTION
# ---------------------------------------------------------

def predict_churn(
    input_data: dict[str, Any],
    model_path: Path | str = MODEL_PATH,
    artifact: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:

    scored = score_dataframe(
        input_df=prepare_input_frame(input_data),
        model_path=model_path,
        artifact=artifact,
        strict=strict,
    )

    row = scored.iloc[0]

    probability = float(row["predicted_probability"])

    return {
        "prediction": int(row["predicted_label"]),
        "prediction_label": str(row["predicted_label_name"]),
        "probability": probability,
        "risk_segment": str(row["risk_segment"]),
        "threshold": float(row["decision_threshold"]),
        "model_name": str(row["model_name"]),
        "probability_percentile": float(row["probability_percentile"]),
        "expected_monthly_revenue_at_risk": float(row["expected_monthly_revenue_at_risk"]),
        "expected_annual_revenue_at_risk": float(row["expected_annual_revenue_at_risk"]),
        "tenure_bucket": str(row.get("tenure_bucket", "")),
        "monthly_charge_bucket": str(row.get("monthly_charge_bucket", "")),
    }


# ---------------------------------------------------------
# SHAP EXPLANATION (FIXED VERSION)
# ---------------------------------------------------------

def get_local_explanation(
    input_data: dict[str, Any],
    model_path: Path | str = MODEL_PATH,
    artifact: dict[str, Any] | None = None,
    top_n: int = 8,
) -> pd.DataFrame:

    resolved_artifact = _resolve_artifact(artifact, model_path)

    preprocessing = resolved_artifact["preprocessing"]
    model = resolved_artifact["model"]

    validated_input = validate_input_frame(
        prepare_input_frame(input_data),
        preprocessing,
        strict=True,
    )

    transformed_input = transform_features(
        validated_input,
        preprocessing,
        strict=True,
    )

    background = resolved_artifact.get("shap_background")

    if background is None:
        raise ValueError("SHAP background data missing in model artifact.")

    try:

        masker = shap.maskers.Independent(background)

        explainer = shap.Explainer(
            lambda x: model.predict_proba(x)[:, 1],
            masker,
        )

        shap_values = explainer(transformed_input)

        contribution_values = shap_values.values[0]

    except Exception:

        contribution_values = [0] * len(transformed_input.columns)

    explanation = pd.DataFrame(
        {
            "feature": transformed_input.columns,
            "display_feature": [
                format_feature_name(f)
                for f in transformed_input.columns
            ],
            "shap_value": contribution_values,
        }
    )

    explanation["impact"] = explanation["shap_value"].apply(
        lambda v: "Increase Churn Risk" if v >= 0 else "Reduce Churn Risk"
    )

    explanation["abs_shap_value"] = explanation["shap_value"].abs()

    return (
        explanation
        .sort_values("abs_shap_value", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# LOCAL TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    artifact = load_model_artifact()

    sample_customer = artifact["preprocessing"]["default_input_values"].copy()

    sample_customer.update(
        {
            "tenure": 8,
            "MonthlyCharges": 89.5,
            "TotalCharges": 716.0,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "TechSupport": "No",
        }
    )

    result = predict_churn(sample_customer, artifact=artifact, strict=True)

    print("Prediction:", result["prediction_label"])
    print("Probability:", f"{result['probability']:.2%}")
    print("Risk Segment:", segment_probability(result["probability"]))
