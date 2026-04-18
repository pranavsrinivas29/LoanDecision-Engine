import json
import os

import requests
import streamlit as st

from utils import get_request, post_request

st.set_page_config(
    page_title="Loan Approval Agent",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------
# Sidebar / Backend config
# ---------------------------------------------------
st.sidebar.title("Backend Settings")

default_api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
api_base_url = st.sidebar.text_input(
    "FastAPI Base URL",
    value=default_api_base_url,
).rstrip("/")

st.sidebar.markdown("---")
st.sidebar.subheader("Health Check")

if st.sidebar.button("Check Backend Health"):
    try:
        health = get_request(api_base_url, "/health")
        st.sidebar.success(
            f"{health.get('status', 'ok')}: {health.get('message', '')}"
        )
    except Exception as e:
        st.sidebar.error(f"Health check failed: {e}")

st.sidebar.markdown("---")
action = st.sidebar.radio(
    "Select Action",
    options=["Predict", "Explain", "Summarize"],
    index=2,
)

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def render_prediction_result(result: dict) -> None:
    prediction_result = result.get("prediction_result", {})

    prediction = prediction_result.get("prediction", "Unknown")
    probability = prediction_result.get("probability")
    threshold = prediction_result.get("threshold")

    if str(prediction).lower() == "approved":
        st.success(f"Prediction: {prediction}")
    else:
        st.error(f"Prediction: {prediction}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Approval Probability",
            f"{probability:.4f}" if probability is not None else "N/A",
        )
    with col2:
        st.metric(
            "Threshold",
            f"{threshold:.2f}" if threshold is not None else "N/A",
        )


def render_local_explanation(result: dict) -> None:
    local_explanation = result.get("local_explanation", {})
    top_positive = local_explanation.get("top_positive", {})
    top_negative = local_explanation.get("top_negative", {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Positive Contributors")
        if top_positive:
            for feature, value in top_positive.items():
                st.write(f"**{feature}**: {value:.4f}")
        else:
            st.info("No positive contributors found.")

    with col2:
        st.subheader("Top Negative Contributors")
        if top_negative:
            for feature, value in top_negative.items():
                st.write(f"**{feature}**: {value:.4f}")
        else:
            st.info("No negative contributors found.")


def render_summary(result: dict) -> None:
    summary = result.get("summary", "")
    if summary:
        st.subheader("Agent Summary")
        st.write(summary)
    else:
        st.info("No summary returned.")


def render_raw_json(result: dict) -> None:
    with st.expander("Show Raw JSON Response"):
        st.json(result)


# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("💳 Loan Approval Agent")
st.write(
    "Enter applicant details below and call the backend for prediction, explanation, or summary."
)

# Optional debug line while Docker testing
st.caption(f"Backend URL in use: {api_base_url}")

# ---------------------------------------------------
# Input Form
# ---------------------------------------------------
with st.form("loan_application_form"):
    st.subheader("Loan Application Input")

    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.number_input(
            "Number of Dependents", min_value=0, value=2, step=1
        )
        education = st.selectbox(
            "Education", options=["Graduate", "Not Graduate"]
        )
        self_employed = st.selectbox(
            "Self Employed", options=["No", "Yes"]
        )
        income_annum = st.number_input(
            "Annual Income", min_value=0.0, value=9600000.0, step=10000.0
        )
        loan_amount = st.number_input(
            "Loan Amount", min_value=0.0, value=29900000.0, step=10000.0
        )
        loan_term = st.number_input(
            "Loan Term", min_value=1.0, value=12.0, step=1.0
        )

    with col2:
        cibil_score = st.number_input(
            "CIBIL Score", min_value=0.0, value=778.0, step=1.0
        )
        residential_assets_value = st.number_input(
            "Residential Assets Value", min_value=0.0, value=2400000.0, step=10000.0
        )
        commercial_assets_value = st.number_input(
            "Commercial Assets Value", min_value=0.0, value=17600000.0, step=10000.0
        )
        luxury_assets_value = st.number_input(
            "Luxury Assets Value", min_value=0.0, value=22700000.0, step=10000.0
        )
        bank_asset_value = st.number_input(
            "Bank Asset Value", min_value=0.0, value=8000000.0, step=10000.0
        )

    submitted = st.form_submit_button(f"Run {action}")

# ---------------------------------------------------
# Payload
# ---------------------------------------------------
payload = {
    "no_of_dependents": int(no_of_dependents),
    "education": education,
    "self_employed": self_employed,
    "income_annum": float(income_annum),
    "loan_amount": float(loan_amount),
    "loan_term": float(loan_term),
    "cibil_score": float(cibil_score),
    "residential_assets_value": float(residential_assets_value),
    "commercial_assets_value": float(commercial_assets_value),
    "luxury_assets_value": float(luxury_assets_value),
    "bank_asset_value": float(bank_asset_value),
}

# ---------------------------------------------------
# Action mapping
# ---------------------------------------------------
endpoint_map = {
    "Predict": "/predict",
    "Explain": "/explain",
    "Summarize": "/summarize",
}

# ---------------------------------------------------
# Run request
# ---------------------------------------------------
if submitted:
    endpoint = endpoint_map[action]

    try:
        with st.spinner(f"Calling {endpoint} ..."):
            result = post_request(api_base_url, endpoint, payload)

        st.success(f"{action} request completed successfully.")

        if action == "Predict":
            render_prediction_result(result)

        elif action == "Explain":
            render_prediction_result(result)
            st.markdown("---")
            render_local_explanation(result)

        elif action == "Summarize":
            render_prediction_result(result)
            st.markdown("---")
            render_local_explanation(result)
            st.markdown("---")
            render_summary(result)

        st.markdown("---")
        render_raw_json(result)

    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error: {e}")
        if e.response is not None:
            try:
                st.json(e.response.json())
            except Exception:
                st.text(e.response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI backend. Make sure the backend is running."
        )

    except requests.exceptions.Timeout:
        st.error("Request timed out.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")

# ---------------------------------------------------
# Footer / Example payload
# ---------------------------------------------------
with st.expander("Show Submitted Payload"):
    st.code(json.dumps(payload, indent=2), language="json")