# Writes model_info.json so the FastAPI service and Streamlit dashboard
# can display metadata without needing to query MLflow directly."""

import json
from datetime import datetime


def save_model_info(model_name: str, metrics: dict, version: str, path: str = "models/model_info.json"):
    info = {
        "model_name": model_name,
        "model_version": str(version),
        "roc_auc": round(metrics["roc_auc"], 4),
        "f1": round(metrics["f1"], 4),
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info