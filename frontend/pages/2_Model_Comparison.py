import streamlit as st
import pandas as pd
import mlflow
import os

st.title("📈 Model Comparison")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "customer_churn"


@st.cache_data(ttl=60)
def load_experiment_runs():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return pd.DataFrame()
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs


runs_df = load_experiment_runs()

if runs_df.empty:
    st.warning("No MLflow runs found. Train models first (Phase 6).")
else:
    display_cols = {
        "tags.mlflow.runName": "Model",
        "metrics.accuracy": "Accuracy",
        "metrics.precision": "Precision",
        "metrics.recall": "Recall",
        "metrics.f1": "F1",
        "metrics.roc_auc": "ROC-AUC",
    }
    available_cols = [c for c in display_cols if c in runs_df.columns]
    table = runs_df[available_cols].rename(columns=display_cols)
    table = table.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

    best_model_name = table.iloc[0]["Model"]
    st.success(f"🏆 Production model: **{best_model_name}**")

    st.dataframe(table.style.highlight_max(subset=["ROC-AUC", "F1"], color="lightgreen"))

    st.subheader("Metric Comparison")
    chart_metric = st.selectbox("Choose metric to chart", ["ROC-AUC", "F1", "Accuracy"])
    st.bar_chart(table.set_index("Model")[chart_metric])