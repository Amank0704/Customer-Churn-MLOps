"""
Manually-triggered retraining workflow:
New Data -> Validate -> Preprocess -> Train Models -> Compare
-> Best Candidate -> Compare with Production -> Promote if Better
"""

import mlflow
import joblib

from src.config import MODEL_DIR
from src.data.load_data import load_raw_data
from src.data.validate_data import run_all_validations
from src.preprocessing.preprocess import get_train_test_split
from src.training.train_with_mlflow import train_and_log_all, REGISTERED_MODEL_NAME
from api.model_loader import load_model_info


def get_current_production_metric() -> float:
    """Read the ROC-AUC of the currently deployed production model."""
    info = load_model_info()
    return info.get("roc_auc", 0.0)


def retrain_and_promote(data_path: str = None):
    print("Step 1: Load new data")
    df = load_raw_data(data_path) if data_path else load_raw_data()

    print("Step 2: Validate")
    run_all_validations(df)

    print("Step 3-5: Preprocess, train, compare models (via existing MLflow training flow)")
    results_df, best_candidate = train_and_log_all()

    print("Step 6: Compare candidate vs current production model")
    current_roc_auc = get_current_production_metric()
    candidate_roc_auc = best_candidate["roc_auc"]

    print(f"Current production ROC-AUC: {current_roc_auc:.4f}")
    print(f"New candidate ROC-AUC:      {candidate_roc_auc:.4f}")

    if candidate_roc_auc > current_roc_auc:
        print("Step 7: Candidate is better. Promoting to production.")
        joblib.dump(best_candidate["pipeline"], f"{MODEL_DIR}best_model.pkl")

        from src.training.save_model_info import save_model_info
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(REGISTERED_MODEL_NAME)[0].version
        save_model_info(best_candidate["model_name"], best_candidate, latest_version)

        print(f"New production model: {best_candidate['model_name']} (version {latest_version})")
        return True
    else:
        print("Step 7: Candidate did NOT outperform production. Keeping current model.")
        return False


if __name__ == "__main__":
    retrain_and_promote()