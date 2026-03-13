from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import enrich_scored_frame, summarize_segments
from src.preprocessing import (
    REPORTS_DIR,
    TARGET_COLUMN,
    TARGET_MAPPING,
    create_preprocessed_datasets,
    transform_features,
)


MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
THRESHOLD_COMPARISON_PATH = PROJECT_ROOT / "reports" / "threshold_comparison.csv"
SEGMENT_KPI_PATH = PROJECT_ROOT / "reports" / "segment_kpis.csv"
PRIORITY_SEGMENT_PATH = PROJECT_ROOT / "reports" / "priority_segments.csv"
ERROR_ANALYSIS_PATH = PROJECT_ROOT / "reports" / "error_analysis.csv"
CALIBRATION_PATH = PROJECT_ROOT / "reports" / "calibration_table.csv"
LIFT_TABLE_PATH = PROJECT_ROOT / "reports" / "lift_table.csv"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "test_predictions.csv"
SHAP_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "shap_feature_importance.csv"

SEGMENT_DIMENSIONS = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "tenure_bucket",
    "monthly_charge_bucket",
]

sns.set_theme(style="whitegrid", palette="crest")


def build_model_candidates(scale_pos_weight: float) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            learning_rate=0.05,
            max_depth=5,
            n_estimators=300,
            subsample=1.0,
            colsample_bytree=1.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_predictions(
    y_true: pd.Series,
    y_probability: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_probability >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "average_precision": average_precision_score(y_true, y_probability),
    }


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
) -> tuple[XGBClassifier, dict[str, Any], float]:
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    base_estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, float(grid_search.best_score_)


def plot_confusion_matrix_figure(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(roc_payload: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for payload in roc_payload:
        ax.plot(
            payload["fpr"],
            payload["tpr"],
            label=f"{payload['model_name']} (AUC={payload['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_recall_figure(
    y_true: pd.Series,
    y_probability: pd.Series,
    output_path: Path,
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    average_precision = average_precision_score(y_true, y_probability)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#157f8a", linewidth=2)
    ax.set_title(f"Precision-Recall Curve (AP={average_precision:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return float(average_precision)


def plot_calibration_figure(
    y_true: pd.Series,
    y_probability: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(y_true, y_probability, n_bins=10, strategy="quantile")
    calibration_df = pd.DataFrame(
        {
            "mean_predicted_probability": prob_pred,
            "observed_positive_rate": prob_true,
        }
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color="#157f8a", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect Calibration")
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return calibration_df


def build_lift_table(
    y_true: pd.Series,
    y_probability: pd.Series,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    scored = pd.DataFrame({"actual_label": y_true, "predicted_probability": y_probability})
    scored = scored.sort_values("predicted_probability", ascending=False).reset_index(drop=True)

    scored["rank"] = np.arange(1, len(scored) + 1)
    scored["population_share"] = scored["rank"] / len(scored)
    scored["decile"] = np.ceil(scored["population_share"] * n_bins).astype(int).clip(upper=n_bins)

    baseline_churn_rate = float(scored["actual_label"].mean())
    total_churners = int(scored["actual_label"].sum())

    grouped = scored.groupby("decile", sort=True)
    lift_table = grouped.agg(
        customer_count=("actual_label", "size"),
        churners=("actual_label", "sum"),
        churn_rate=("actual_label", "mean"),
        avg_predicted_probability=("predicted_probability", "mean"),
    ).reset_index()
    lift_table["lift_vs_baseline"] = lift_table["churn_rate"] / baseline_churn_rate
    lift_table["cumulative_churners"] = lift_table["churners"].cumsum()
    lift_table["cumulative_capture_rate"] = lift_table["cumulative_churners"] / total_churners
    lift_table["cumulative_population_share"] = lift_table["customer_count"].cumsum() / len(scored)

    top_decile_capture = float(lift_table.loc[lift_table["decile"] == 1, "cumulative_capture_rate"].iloc[0])
    return lift_table, top_decile_capture


def plot_lift_figure(lift_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(lift_table["decile"].astype(str), lift_table["lift_vs_baseline"], color="#157f8a")
    axes[0].set_title("Lift by Decile")
    axes[0].set_xlabel("Decile")
    axes[0].set_ylabel("Lift vs Baseline")

    axes[1].plot(
        lift_table["cumulative_population_share"],
        lift_table["cumulative_capture_rate"],
        marker="o",
        color="#0f766e",
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[1].set_title("Cumulative Churn Capture")
    axes[1].set_xlabel("Population Share")
    axes[1].set_ylabel("Capture Rate")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def extract_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    ).sort_values("importance", ascending=False)
    return importance_df.reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
    title: str = "Top Feature Importance",
) -> None:
    top_features = importance_df.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features["feature"], top_features["importance"], color="#157f8a")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_model_comparison(model_comparison: pd.DataFrame, output_path: Path) -> None:
    metric_frame = model_comparison.melt(
        id_vars="model_name",
        value_vars=[
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "average_precision",
        ],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=metric_frame, x="metric", y="score", hue="model_name", ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def compute_shap_importance(
    model: Any,
    background: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame:
    explainer = shap.Explainer(model, background)
    shap_values = explainer(sample)
    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[:, :, 1]

    shap_importance = pd.DataFrame(
        {
            "feature": sample.columns,
            "importance": np.abs(values).mean(axis=0),
        }
    ).sort_values("importance", ascending=False)
    return shap_importance.reset_index(drop=True)


def plot_segment_revenue_at_risk(segment_summary: pd.DataFrame, output_path: Path) -> None:
    top_segments = segment_summary.head(10).copy()
    top_segments["segment_label"] = top_segments["dimension"] + ": " + top_segments["segment"]
    top_segments = top_segments.sort_values("expected_monthly_revenue_at_risk")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top_segments["segment_label"],
        top_segments["expected_monthly_revenue_at_risk"],
        color="#157f8a",
    )
    ax.set_title("Top Segments by Expected Monthly Revenue at Risk")
    ax.set_xlabel("Expected Monthly Revenue at Risk")
    ax.set_ylabel("Segment")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_error_analysis(scored_test_df: pd.DataFrame) -> pd.DataFrame:
    error_summary = summarize_segments(scored_test_df, SEGMENT_DIMENSIONS)
    if error_summary.empty:
        return error_summary

    return error_summary[
        [
            "dimension",
            "segment",
            "customer_count",
            "actual_churn_rate",
            "avg_predicted_probability",
            "false_negative_count",
            "false_negative_rate",
            "false_positive_count",
            "false_positive_rate",
            "expected_monthly_revenue_at_risk",
        ]
    ].sort_values(
        ["false_negative_rate", "expected_monthly_revenue_at_risk"],
        ascending=False,
    ).reset_index(drop=True)


def plot_error_analysis(error_summary: pd.DataFrame, output_path: Path) -> None:
    display_df = error_summary.copy()
    display_df["segment_label"] = display_df["dimension"] + ": " + display_df["segment"]

    fn_df = display_df.head(10).sort_values("false_negative_rate")
    fp_df = display_df.sort_values("false_positive_rate", ascending=False).head(10).sort_values(
        "false_positive_rate"
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].barh(fn_df["segment_label"], fn_df["false_negative_rate"], color="#b91c1c")
    axes[0].set_title("Segments with Highest False Negative Rate")
    axes[0].set_xlabel("False Negative Rate")
    axes[0].set_ylabel("Segment")

    axes[1].barh(fp_df["segment_label"], fp_df["false_positive_rate"], color="#d97706")
    axes[1].set_title("Segments with Highest False Positive Rate")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("Segment")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train_models() -> dict[str, Any]:
    datasets = create_preprocessed_datasets()
    clean_df = datasets["clean_df"]
    X_train_raw = datasets["X_train_raw"]
    X_test_raw = datasets["X_test_raw"]
    X_train = datasets["X_train"]
    X_test = datasets["X_test"]
    y_train = datasets["y_train"]
    y_test = datasets["y_test"]
    preprocessing_artifacts = datasets["preprocessing_artifacts"]

    positive_class_count = int(y_train.sum())
    negative_class_count = int((1 - y_train).sum())
    scale_pos_weight = negative_class_count / positive_class_count

    models = build_model_candidates(scale_pos_weight)
    tuned_xgb_model, best_params, best_cv_score = tune_xgboost(
        X_train,
        y_train,
        scale_pos_weight,
    )
    models["XGBoost"] = tuned_xgb_model

    evaluation_rows: list[dict[str, Any]] = []
    roc_payload: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_probability = pd.Series(model.predict_proba(X_test)[:, 1], dtype=float)
        metrics = evaluate_predictions(y_test, y_probability, threshold=0.5)
        fpr, tpr, _ = roc_curve(y_test, y_probability)

        evaluation_rows.append({"model_name": model_name, **metrics})
        roc_payload.append(
            {
                "model_name": model_name,
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc": metrics["roc_auc"],
            }
        )
        fitted_models[model_name] = model

    model_comparison = pd.DataFrame(evaluation_rows).sort_values("roc_auc", ascending=False)
    model_comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    plot_model_comparison(model_comparison, REPORTS_DIR / "model_comparison.png")
    plot_roc_curves(roc_payload, REPORTS_DIR / "roc_curve_comparison.png")

    best_model_name = model_comparison.iloc[0]["model_name"]
    best_model = fitted_models[best_model_name]
    best_probability = pd.Series(best_model.predict_proba(X_test)[:, 1], dtype=float)

    threshold_rows = []
    for threshold in [0.5, 0.4]:
        threshold_metrics = evaluate_predictions(y_test, best_probability, threshold=threshold)
        threshold_rows.append({"threshold": threshold, **threshold_metrics})
    threshold_comparison = pd.DataFrame(threshold_rows)
    threshold_comparison.to_csv(THRESHOLD_COMPARISON_PATH, index=False)

    decision_threshold = 0.40
    y_pred_threshold = (best_probability >= decision_threshold).astype(int)
    plot_confusion_matrix_figure(
        y_test,
        y_pred_threshold,
        f"{best_model_name} Confusion Matrix @ {decision_threshold:.2f}",
        REPORTS_DIR / "confusion_matrix.png",
    )

    average_precision = plot_precision_recall_figure(
        y_test,
        best_probability,
        REPORTS_DIR / "precision_recall_curve.png",
    )
    calibration_df = plot_calibration_figure(
        y_test,
        best_probability,
        REPORTS_DIR / "calibration_curve.png",
    )
    calibration_df.to_csv(CALIBRATION_PATH, index=False)

    lift_table, top_decile_capture = build_lift_table(y_test, best_probability)
    lift_table.to_csv(LIFT_TABLE_PATH, index=False)
    plot_lift_figure(lift_table, REPORTS_DIR / "lift_analysis.png")

    test_scored_df = enrich_scored_frame(
        X_test_raw,
        best_probability,
        threshold=decision_threshold,
        actual=y_test,
    )
    test_scored_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    error_analysis = build_error_analysis(test_scored_df)
    error_analysis.to_csv(ERROR_ANALYSIS_PATH, index=False)
    plot_error_analysis(error_analysis, REPORTS_DIR / "error_analysis.png")

    raw_population_features = clean_df.drop(columns=[TARGET_COLUMN])
    population_targets = clean_df[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)
    population_transformed = transform_features(
        raw_population_features,
        preprocessing_artifacts,
        strict=True,
    )
    population_probability = pd.Series(
        best_model.predict_proba(population_transformed)[:, 1],
        dtype=float,
    )
    scored_population_df = enrich_scored_frame(
        raw_population_features,
        population_probability,
        threshold=decision_threshold,
        actual=population_targets,
    )

    segment_kpis = summarize_segments(scored_population_df, SEGMENT_DIMENSIONS)
    segment_kpis.to_csv(SEGMENT_KPI_PATH, index=False)
    priority_segments = segment_kpis.sort_values(
        ["expected_monthly_revenue_at_risk", "actual_churn_rate"],
        ascending=False,
    ).head(15)
    priority_segments.to_csv(PRIORITY_SEGMENT_PATH, index=False)
    plot_segment_revenue_at_risk(
        priority_segments,
        REPORTS_DIR / "segment_revenue_at_risk.png",
    )

    feature_importance = extract_feature_importance(
        best_model,
        preprocessing_artifacts["feature_columns"],
    )
    feature_importance.to_csv(PROJECT_ROOT / "reports" / "feature_importance.csv", index=False)
    plot_feature_importance(
        feature_importance,
        REPORTS_DIR / "feature_importance.png",
    )

    shap_background = X_train.sample(min(200, len(X_train)), random_state=42).reset_index(drop=True)
    shap_sample = X_test.sample(min(400, len(X_test)), random_state=42).reset_index(drop=True)
    shap_importance = compute_shap_importance(best_model, shap_background, shap_sample)
    shap_importance.to_csv(SHAP_IMPORTANCE_PATH, index=False)
    plot_feature_importance(
        shap_importance.rename(columns={"importance": "importance"}),
        REPORTS_DIR / "shap_feature_importance.png",
        title="Top SHAP Feature Importance",
    )

    portfolio_summary = {
        "customer_count": int(len(scored_population_df)),
        "average_predicted_probability": float(scored_population_df["predicted_probability"].mean()),
        "high_risk_customer_share": float((scored_population_df["risk_segment"] == "High Risk").mean()),
        "expected_monthly_revenue_at_risk": float(
            scored_population_df["expected_monthly_revenue_at_risk"].sum()
        ),
        "expected_annual_revenue_at_risk": float(
            scored_population_df["expected_annual_revenue_at_risk"].sum()
        ),
    }

    model_artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "preprocessing": preprocessing_artifacts,
        "metrics": model_comparison.to_dict(orient="records"),
        "threshold_metrics": threshold_comparison.to_dict(orient="records"),
        "decision_threshold": decision_threshold,
        "feature_importance": feature_importance.head(20).to_dict(orient="records"),
        "shap_feature_importance": shap_importance.head(20).to_dict(orient="records"),
        "shap_background": shap_background,
        "xgboost_best_params": best_params,
        "xgboost_best_cv_score": best_cv_score,
        "target_column": TARGET_COLUMN,
        "reference_customer_scores": scored_population_df["predicted_probability"].round(8).tolist(),
        "segment_kpis": segment_kpis.to_dict(orient="records"),
        "portfolio_summary": portfolio_summary,
        "top_decile_capture": top_decile_capture,
        "average_precision": average_precision,
        "brier_score": float(brier_score_loss(y_test, best_probability)),
        "calibration_table": calibration_df.to_dict(orient="records"),
        "lift_table": lift_table.to_dict(orient="records"),
        "error_analysis": error_analysis.head(20).to_dict(orient="records"),
    }
    joblib.dump(model_artifact, MODEL_PATH)

    return {
        "model_artifact": model_artifact,
        "model_comparison": model_comparison,
        "threshold_comparison": threshold_comparison,
        "scale_pos_weight": scale_pos_weight,
        "segment_kpis": segment_kpis,
        "error_analysis": error_analysis,
    }


if __name__ == "__main__":
    training_outputs = train_models()
    print("Best model:", training_outputs["model_artifact"]["model_name"])
    print("Scale pos weight:", round(training_outputs["scale_pos_weight"], 3))
    print("Top decile capture:", f"{training_outputs['model_artifact']['top_decile_capture']:.2%}")
    print(training_outputs["model_comparison"].to_string(index=False))
