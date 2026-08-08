import joblib
import json
import os

MODEL_PATH = "models/best_model.pkl"
MODEL_INFO_PATH = "models/model_info.json"

_model = None
_model_info = None


def load_model():
    """Load the trained pipeline once and cache it in memory.
    The API never retrains — it only loads an already-trained artifact."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run training first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def load_model_info():
    """Load model metadata (name, version, metrics) saved alongside the model."""
    global _model_info
    if _model_info is None:
        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, "r") as f:
                _model_info = json.load(f)
        else:
            _model_info = {
                "model_name": "unknown",
                "model_version": "1",
                "roc_auc": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
                "training_date": "unknown"
            }
    return _model_info