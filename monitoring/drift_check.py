import json
import pandas as pd
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.config import RAW_DATA_PATH
from src.data.load_data import load_raw_data
from src.preprocessing.preprocess import clean_raw_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES

MIN_PRODUCTION_ROWS = 30


def build_reference_data() -> pd.DataFrame:
    # Use the original training data as the reference distribution."""
    df = load_raw_data(RAW_DATA_PATH)
    df = clean_raw_data(df)
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_current_data(logs_df: pd.DataFrame) -> pd.DataFrame:
  # Parse logged input_features JSON strings into a DataFrame matching reference columns."""
    parsed_rows = [json.loads(row) for row in logs_df["input_features"]]
    current_df = pd.DataFrame(parsed_rows)
    current_df["TotalCharges"] = pd.to_numeric(current_df["TotalCharges"], errors="coerce")
    return current_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def run_drift_check(logs_df: pd.DataFrame) -> dict:
    """Compare reference (training) data vs. recent production inputs for drift."""
    if len(logs_df) < MIN_PRODUCTION_ROWS:
        return {
            "dataset_drift": False,
            "drifted_features": 0,
            "message": "Insufficient production data for drift analysis.",
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    reference_df = build_reference_data()
    current_df = build_current_data(logs_df)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    result = report.as_dict()

    drift_metrics = result["metrics"][0]["result"]

    return {
        "dataset_drift": drift_metrics["dataset_drift"],
        "drifted_features": drift_metrics["number_of_drifted_columns"],
        "total_features": drift_metrics["number_of_columns"],
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


if __name__ == "__main__":
    from src.database.db import SessionLocal
    from src.database.models import PredictionLog

    db = SessionLocal()
    logs = db.query(PredictionLog).all()
    db.close()

    logs_df = pd.DataFrame([{
        "input_features": log.input_features
    } for log in logs])

    if logs_df.empty:
        print("No logs found.")
    else:
        result = run_drift_check(logs_df)
        print(result)