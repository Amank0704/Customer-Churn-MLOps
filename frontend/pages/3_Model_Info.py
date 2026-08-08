import streamlit as st
from utils import call_model_info

st.title("ℹ️ Production Model Information")

try:
    info = call_model_info()

    col1, col2 = st.columns(2)
    col1.metric("Model Name", info["model_name"])
    col2.metric("Version", info["model_version"])

    st.subheader("Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", f"{info['roc_auc']:.4f}")
    col2.metric("F1 Score", f"{info['f1']:.4f}")
    col3.metric("Accuracy", f"{info['accuracy']:.4f}")

    st.caption(f"Trained on: {info['training_date']}")

except Exception as e:
    st.error(f"Could not load model info: {e}")