import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
Welcome to the Customer Churn MLOps dashboard.

Use the sidebar to navigate between:

- **Prediction** — Predict churn for a single customer
- **Model Comparison** — Compare all trained models
- **Model Info** — View the current production model's details
- **Monitoring** — Track prediction volume and data drift
""")