#v2 made changes in path
import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.title("📈 Model Comparison")

MODEL_INFO_PATH = Path("models/model_info.json")


@st.cache_data(ttl=60)
def load_model_comparison():
    if not MODEL_INFO_PATH.exists():
        return pd.DataFrame()

    with open(MODEL_INFO_PATH, "r") as f:
        info = json.load(f)

    comparison = info.get("model_comparison", [])

    if not comparison:
        return pd.DataFrame()

    return pd.DataFrame(comparison)


runs_df = load_model_comparison()

if runs_df.empty:
    st.warning(
        "No model comparison results found. "
        "Run the training pipeline first."
    )
else:
    display_cols = {
        "model_name": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    }

    table = runs_df[
        [c for c in display_cols if c in runs_df.columns]
    ].rename(columns=display_cols)

    table = table.sort_values(
        by="ROC-AUC",
        ascending=False
    ).reset_index(drop=True)

    best_model_name = table.iloc[0]["Model"]

    st.success(
        f"🏆 Production model: **{best_model_name}**"
    )

    st.dataframe(
        table.style.highlight_max(
            subset=["ROC-AUC", "F1"],
            color="lightgreen"
        ),
        use_container_width=True
    )

    st.subheader("Metric Comparison")

    chart_metric = st.selectbox(
        "Choose metric to chart",
        ["ROC-AUC", "F1", "Accuracy"]
    )

    st.bar_chart(
        table.set_index("Model")[chart_metric]
    )