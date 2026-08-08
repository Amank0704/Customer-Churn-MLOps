import json
from src.database.db import SessionLocal
from src.database.models import PredictionLog


def save_prediction_log(model_version: str, input_features: dict, prediction: str, probability: float):
    """Insert one prediction record into the database."""
    db = SessionLocal()
    try:
        log_entry = PredictionLog(
            model_version=model_version,
            input_features=json.dumps(input_features),
            prediction=prediction,
            probability=probability
        )
        db.add(log_entry)
        db.commit()
    finally:
        db.close()