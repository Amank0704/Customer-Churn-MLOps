import streamlit as st
import pandas as pd
import json
import os
from sqlalchemy import create_engine, text

st.title("📡 Monitoring")

from dotenv import load_dotenv

load_dotenv()

# DATABASE_URL = (
#     f"postgresql://{os.getenv('POSTGRES_USER')}:"
#     f"{os.getenv('POSTGRES_PASSWORD')}@"
#     f"{os.getenv('POSTGRES_HOST')}:"
#     f"{os.getenv('POSTGRES_PORT')}/"
#     f"{os.getenv('POSTGRES_DB')}"
# )

# engine = create_engine(DATABASE_URL)
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

engine = create_engine(DATABASE_URL)


@st.cache_data(ttl=30)
def load_logs():
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT timestamp, model_version, input_features, prediction, probability "
            "FROM prediction_logs ORDER BY timestamp DESC"
        ))
        rows = result.fetchall()
    return pd.DataFrame(rows, columns=["timestamp", "model_version", "input_features", "prediction", "probability"])


logs_df = load_logs()

if logs_df.empty:
    st.info("No predictions logged yet. Make some predictions on the Prediction page first.")
else:
    st.subheader("Prediction Volume")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", len(logs_df))
    col2.metric("Churn Predicted", (logs_df["prediction"] == "Churn").sum())
    col3.metric("Avg Confidence", f"{logs_df['probability'].mean():.2%}")

    st.subheader("Churn vs Stay Distribution")
    st.bar_chart(logs_df["prediction"].value_counts())

    st.subheader("Drift Status")
    if len(logs_df) < 30:
        st.warning("Insufficient production data for drift analysis.")
    else:
        from monitoring.drift_check import run_drift_check
        try:
            drift_result = run_drift_check(logs_df)
            if drift_result["dataset_drift"]:
                st.error(f"⚠️ Drift detected in {drift_result['drifted_features']} feature(s).")
            else:
                st.success("✅ No significant drift detected.")
            st.caption(f"Last checked: {drift_result['checked_at']}")
        except Exception as e:
            st.error(f"Drift check failed: {e}")

    with st.expander("Raw prediction logs"):
        st.dataframe(logs_df)