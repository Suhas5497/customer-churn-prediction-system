from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import get_local_explanation, load_model_artifact, predict_batch, predict_churn


st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="C",
    layout="wide",
)


@st.cache_resource
def get_artifact():
    return load_model_artifact()


def build_recommendations(input_payload: dict[str, object], probability: float) -> list[str]:
    recommendations = []

    if input_payload["Contract"] == "Month-to-month":
        recommendations.append("Promote a discounted long-term contract to reduce month-to-month churn risk.")
    if input_payload["TechSupport"] == "No":
        recommendations.append("Offer a technical support bundle to reduce service friction.")
    if input_payload["InternetService"] == "Fiber optic":
        recommendations.append("Review price-to-value perception and service quality for fiber customers.")
    if input_payload["PaymentMethod"] == "Electronic check":
        recommendations.append("Move the customer toward a lower-friction automatic payment method.")
    if probability >= 0.65:
        recommendations.append("Escalate to a high-priority retention workflow this billing cycle.")
    elif probability >= 0.35:
        recommendations.append("Add to a mid-risk nurture segment with proactive outreach and incentives.")
    else:
        recommendations.append("Maintain the current journey and monitor pricing or tenure changes over time.")

    return recommendations[:4]


def risk_badge_html(risk_segment: str) -> str:
    badge_class = {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
    }[risk_segment]
    return f'<span class="risk-badge {badge_class}">{risk_segment}</span>'


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def get_segment_benchmark(
    segment_kpis_df: pd.DataFrame,
    dimension: str,
    segment: str,
) -> pd.Series | None:
    benchmark = segment_kpis_df[
        (segment_kpis_df["dimension"] == dimension)
        & (segment_kpis_df["segment"].astype(str) == str(segment))
    ]
    if benchmark.empty:
        return None
    return benchmark.iloc[0]


def build_template_file(preprocessing: dict[str, object]) -> bytes:
    template_df = pd.DataFrame([preprocessing["default_input_values"]])
    buffer = BytesIO()
    template_df.to_csv(buffer, index=False)
    return buffer.getvalue()


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(4, 120, 87, 0.12), transparent 28%),
            linear-gradient(180deg, #f5f7f2 0%, #eef3ec 100%);
    }
    .hero-card, .panel-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.07);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #0f172a;
    }
    .hero-copy {
        color: #334155;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .risk-low {
        color: #166534;
        background: #dcfce7;
    }
    .risk-medium {
        color: #92400e;
        background: #fef3c7;
    }
    .risk-high {
        color: #991b1b;
        background: #fee2e2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    artifact = get_artifact()
except Exception as exc:
    st.error("Model artifact not found or invalid.")
    st.code(f"python src/train_model.py\n\n{exc}")
    st.stop()

preprocessing = artifact["preprocessing"]
default_values = preprocessing["default_input_values"]
category_levels = preprocessing["category_levels"]
metrics_df = pd.DataFrame(artifact["metrics"]).round(3)
feature_importance_df = pd.DataFrame(artifact["feature_importance"])
shap_importance_df = pd.DataFrame(artifact.get("shap_feature_importance", []))
segment_kpis_df = pd.DataFrame(artifact.get("segment_kpis", []))
portfolio_summary = artifact.get("portfolio_summary", {})

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Customer Churn Prediction System</div>
        <p class="hero-copy">
            Analyst-ready churn intelligence dashboard with portfolio scoring, revenue-at-risk estimates,
            segment benchmarking, and model explainability.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

sidebar_col, main_col = st.columns([0.95, 2.05], gap="large")

with sidebar_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Model Snapshot")
    st.write(f"Selected model: `{artifact['model_name']}`")
    st.write(f"Decision threshold: `{artifact['decision_threshold']:.2f}`")
    st.write(f"Average precision: `{artifact.get('average_precision', 0.0):.3f}`")
    st.write(f"Brier score: `{artifact.get('brier_score', 0.0):.3f}`")
    st.write(f"Top-decile churn capture: `{artifact.get('top_decile_capture', 0.0):.1%}`")
    st.write("Model comparison")
    st.dataframe(
        metrics_df[
            ["model_name", "roc_auc", "average_precision", "recall", "precision", "f1_score"]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.write("Portfolio summary")
    if portfolio_summary:
        st.metric("Expected Monthly Revenue at Risk", format_currency(portfolio_summary["expected_monthly_revenue_at_risk"]))
        st.metric("High-Risk Customer Share", f"{portfolio_summary['high_risk_customer_share']:.1%}")

    st.write("Top SHAP drivers")
    if not shap_importance_df.empty:
        st.dataframe(shap_importance_df.head(8), hide_index=True, use_container_width=True)
    elif not feature_importance_df.empty:
        st.dataframe(feature_importance_df.head(8), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with main_col:
    tab_single, tab_batch, tab_insights = st.tabs(
        ["Single Customer", "Batch Scoring", "Portfolio Insights"]
    )

    with tab_single:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Score a Customer")
        st.caption(
            "Use this workflow for individual retention decisions. Core commercial and service inputs are surfaced directly, with the full raw schema available in the advanced panel."
        )

        with st.form("single_customer_form"):
            input_payload = default_values.copy()
            col1, col2, col3 = st.columns(3)

            with col1:
                input_payload["tenure"] = st.slider(
                    "Tenure (months)",
                    min_value=0,
                    max_value=72,
                    value=int(default_values["tenure"]),
                )
                input_payload["MonthlyCharges"] = st.number_input(
                    "Monthly Charges",
                    min_value=0.0,
                    max_value=200.0,
                    value=float(default_values["MonthlyCharges"]),
                    step=1.0,
                )
                input_payload["TotalCharges"] = st.number_input(
                    "Total Charges",
                    min_value=0.0,
                    max_value=10000.0,
                    value=float(default_values["TotalCharges"]),
                    step=10.0,
                )

            with col2:
                input_payload["Contract"] = st.selectbox(
                    "Contract Type",
                    options=category_levels["Contract"],
                    index=category_levels["Contract"].index(default_values["Contract"]),
                )
                input_payload["InternetService"] = st.selectbox(
                    "Internet Service",
                    options=category_levels["InternetService"],
                    index=category_levels["InternetService"].index(default_values["InternetService"]),
                )
                input_payload["TechSupport"] = st.selectbox(
                    "Tech Support",
                    options=category_levels["TechSupport"],
                    index=category_levels["TechSupport"].index(default_values["TechSupport"]),
                )

            with col3:
                input_payload["SeniorCitizen"] = st.selectbox(
                    "Senior Citizen",
                    options=[0, 1],
                    index=int(default_values["SeniorCitizen"]),
                )
                input_payload["PaymentMethod"] = st.selectbox(
                    "Payment Method",
                    options=category_levels["PaymentMethod"],
                    index=category_levels["PaymentMethod"].index(default_values["PaymentMethod"]),
                )
                input_payload["PaperlessBilling"] = st.selectbox(
                    "Paperless Billing",
                    options=category_levels["PaperlessBilling"],
                    index=category_levels["PaperlessBilling"].index(default_values["PaperlessBilling"]),
                )

            with st.expander("Advanced customer attributes"):
                adv1, adv2, adv3 = st.columns(3)

                with adv1:
                    input_payload["gender"] = st.selectbox(
                        "Gender",
                        options=category_levels["gender"],
                        index=category_levels["gender"].index(default_values["gender"]),
                    )
                    input_payload["Partner"] = st.selectbox(
                        "Partner",
                        options=category_levels["Partner"],
                        index=category_levels["Partner"].index(default_values["Partner"]),
                    )
                    input_payload["Dependents"] = st.selectbox(
                        "Dependents",
                        options=category_levels["Dependents"],
                        index=category_levels["Dependents"].index(default_values["Dependents"]),
                    )

                with adv2:
                    input_payload["PhoneService"] = st.selectbox(
                        "Phone Service",
                        options=category_levels["PhoneService"],
                        index=category_levels["PhoneService"].index(default_values["PhoneService"]),
                    )
                    input_payload["MultipleLines"] = st.selectbox(
                        "Multiple Lines",
                        options=category_levels["MultipleLines"],
                        index=category_levels["MultipleLines"].index(default_values["MultipleLines"]),
                    )
                    input_payload["OnlineSecurity"] = st.selectbox(
                        "Online Security",
                        options=category_levels["OnlineSecurity"],
                        index=category_levels["OnlineSecurity"].index(default_values["OnlineSecurity"]),
                    )

                with adv3:
                    input_payload["OnlineBackup"] = st.selectbox(
                        "Online Backup",
                        options=category_levels["OnlineBackup"],
                        index=category_levels["OnlineBackup"].index(default_values["OnlineBackup"]),
                    )
                    input_payload["DeviceProtection"] = st.selectbox(
                        "Device Protection",
                        options=category_levels["DeviceProtection"],
                        index=category_levels["DeviceProtection"].index(default_values["DeviceProtection"]),
                    )
                    input_payload["StreamingTV"] = st.selectbox(
                        "Streaming TV",
                        options=category_levels["StreamingTV"],
                        index=category_levels["StreamingTV"].index(default_values["StreamingTV"]),
                    )
                    input_payload["StreamingMovies"] = st.selectbox(
                        "Streaming Movies",
                        options=category_levels["StreamingMovies"],
                        index=category_levels["StreamingMovies"].index(default_values["StreamingMovies"]),
                    )

            submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            prediction = predict_churn(input_payload, artifact=artifact, strict=True)
            local_explanation = get_local_explanation(input_payload, artifact=artifact)
            positive_drivers = local_explanation[local_explanation["shap_value"] >= 0]
            protective_drivers = local_explanation[local_explanation["shap_value"] < 0]

            contract_benchmark = get_segment_benchmark(segment_kpis_df, "Contract", input_payload["Contract"])
            tenure_benchmark = get_segment_benchmark(segment_kpis_df, "tenure_bucket", prediction["tenure_bucket"])
            internet_benchmark = get_segment_benchmark(
                segment_kpis_df,
                "InternetService",
                input_payload["InternetService"],
            )

            top_metrics = st.columns(4)
            top_metrics[0].metric("Churn Probability", f"{prediction['probability']:.1%}")
            top_metrics[1].metric("Risk Percentile", f"{prediction['probability_percentile']:.1f}")
            top_metrics[2].metric(
                "Expected Monthly Revenue at Risk",
                format_currency(prediction["expected_monthly_revenue_at_risk"]),
            )
            top_metrics[3].metric(
                "Expected Annual Revenue at Risk",
                format_currency(prediction["expected_annual_revenue_at_risk"]),
            )

            result_col, explain_col = st.columns([1.15, 1], gap="large")

            with result_col:
                st.markdown('<div class="panel-card">', unsafe_allow_html=True)
                st.subheader("Prediction Result")
                st.markdown(risk_badge_html(prediction["risk_segment"]), unsafe_allow_html=True)
                st.write(
                    f"The model predicts **{prediction['prediction_label']}** using a production threshold of `{prediction['threshold']:.2f}`."
                )
                st.write("Recommended actions")
                for recommendation in build_recommendations(input_payload, prediction["probability"]):
                    st.write(f"- {recommendation}")

                st.write("Segment benchmarks")
                if contract_benchmark is not None:
                    st.write(
                        f"- Contract benchmark `{input_payload['Contract']}`: actual churn rate `{contract_benchmark['actual_churn_rate']:.1%}`, average model risk `{contract_benchmark['avg_predicted_probability']:.1%}`."
                    )
                if tenure_benchmark is not None:
                    st.write(
                        f"- Tenure bucket `{prediction['tenure_bucket']}`: actual churn rate `{tenure_benchmark['actual_churn_rate']:.1%}`, expected monthly revenue at risk `{format_currency(tenure_benchmark['expected_monthly_revenue_at_risk'])}`."
                    )
                if internet_benchmark is not None:
                    st.write(
                        f"- Internet service `{input_payload['InternetService']}`: actual churn rate `{internet_benchmark['actual_churn_rate']:.1%}`, high-risk share `{internet_benchmark['high_risk_rate']:.1%}`."
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            with explain_col:
                st.markdown('<div class="panel-card">', unsafe_allow_html=True)
                st.subheader("Local Explanation")
                st.caption("SHAP contributions for this customer prediction.")
                st.bar_chart(
                    local_explanation.set_index("display_feature")["shap_value"],
                    use_container_width=True,
                )
                if not positive_drivers.empty:
                    st.write("Drivers increasing risk")
                    st.dataframe(
                        positive_drivers[["display_feature", "shap_value"]].head(5),
                        hide_index=True,
                        use_container_width=True,
                    )
                if not protective_drivers.empty:
                    st.write("Drivers reducing risk")
                    st.dataframe(
                        protective_drivers[["display_feature", "shap_value"]].head(5),
                        hide_index=True,
                        use_container_width=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

    with tab_batch:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Batch Portfolio Scoring")
        st.caption(
            "Upload a CSV with raw customer columns to score an entire portfolio. Missing optional fields will be filled from the training-data defaults."
        )
        st.download_button(
            "Download CSV template",
            data=build_template_file(preprocessing),
            file_name="customer_churn_template.csv",
            mime="text/csv",
            use_container_width=False,
        )
        uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

        if uploaded_file is not None:
            batch_input_df = pd.read_csv(uploaded_file)
            batch_results_df = predict_batch(batch_input_df, artifact=artifact, strict=False)

            summary_cols = st.columns(4)
            summary_cols[0].metric("Rows Scored", f"{len(batch_results_df):,}")
            summary_cols[1].metric(
                "High-Risk Customers",
                f"{int((batch_results_df['risk_segment'] == 'High Risk').sum()):,}",
            )
            summary_cols[2].metric(
                "Average Churn Probability",
                f"{batch_results_df['predicted_probability'].mean():.1%}",
            )
            summary_cols[3].metric(
                "Expected Monthly Revenue at Risk",
                format_currency(batch_results_df["expected_monthly_revenue_at_risk"].sum()),
            )

            risk_counts = (
                batch_results_df["risk_segment"]
                .value_counts()
                .reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
            )
            chart_col, preview_col = st.columns([1, 1.2], gap="large")

            with chart_col:
                st.write("Risk distribution")
                st.bar_chart(risk_counts, use_container_width=True)

            with preview_col:
                st.write("Highest-risk customers")
                preview_df = batch_results_df.sort_values("predicted_probability", ascending=False).head(10)
                st.dataframe(
                    preview_df[
                        [
                            "Contract",
                            "InternetService",
                            "tenure",
                            "MonthlyCharges",
                            "predicted_probability",
                            "risk_segment",
                            "expected_monthly_revenue_at_risk",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

            csv_buffer = BytesIO()
            batch_results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download scored portfolio",
                data=csv_buffer.getvalue(),
                file_name="scored_customer_churn_portfolio.csv",
                mime="text/csv",
            )
        else:
            st.write("Expected raw input columns")
            st.dataframe(
                pd.DataFrame({"column_name": preprocessing["input_columns"]}),
                hide_index=True,
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_insights:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Portfolio Insights")
        st.caption("Analyst-oriented segment views from the scored reference population.")

        insight_cols = st.columns(3)
        insight_cols[0].metric(
            "Reference Monthly Revenue at Risk",
            format_currency(portfolio_summary.get("expected_monthly_revenue_at_risk", 0.0)),
        )
        insight_cols[1].metric(
            "Reference Annual Revenue at Risk",
            format_currency(portfolio_summary.get("expected_annual_revenue_at_risk", 0.0)),
        )
        insight_cols[2].metric(
            "Average Portfolio Risk",
            f"{portfolio_summary.get('average_predicted_probability', 0.0):.1%}",
        )

        if not segment_kpis_df.empty:
            top_segments = segment_kpis_df.sort_values(
                ["expected_monthly_revenue_at_risk", "actual_churn_rate"],
                ascending=False,
            ).head(12)
            st.write("Priority segments")
            st.dataframe(
                top_segments[
                    [
                        "dimension",
                        "segment",
                        "customer_count",
                        "actual_churn_rate",
                        "avg_predicted_probability",
                        "expected_monthly_revenue_at_risk",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.write("Top global churn drivers")
            chart_source = shap_importance_df if not shap_importance_df.empty else feature_importance_df
            if not chart_source.empty:
                st.bar_chart(
                    chart_source.head(10).set_index("feature")["importance"],
                    use_container_width=True,
                )

        st.write("Business interpretation")
        st.write("- Target month-to-month and fiber-optic customers with pricing and contract offers.")
        st.write("- Use expected revenue at risk to prioritize outreach queues, not only probability.")
        st.write("- High churn probability combined with electronic check and missing tech support is a strong intervention segment.")
        st.markdown("</div>", unsafe_allow_html=True)
