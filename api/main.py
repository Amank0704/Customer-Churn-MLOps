import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import CustomerInput, PredictionResponse, ModelInfoResponse
from api.model_loader import load_model, load_model_info
from src.database.log_prediction import save_prediction_log
from src.database.db import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_api")

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # Ensure prediction_logs table exists. Does NOT retrain or load the model here.
    init_db()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    info = load_model_info()
    return info


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        model = load_model()
        info = load_model_info()

        input_dict = customer.model_dump()
        input_df = pd.DataFrame([input_dict])
        proba = model.predict_proba(input_df)[0][1]
        prediction = "Churn" if proba >= 0.5 else "Stay"

        logger.info(f"Prediction made: {prediction} (proba={proba:.4f})")

        save_prediction_log(
            model_version=info["model_version"],
            input_features=input_dict,
            prediction=prediction,
            probability=round(float(proba), 4)
        )

        return PredictionResponse(
            prediction=prediction,
            probability=round(float(proba), 4),
            model_version=info["model_version"]
        )

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=503, detail="Model not available. Train a model first.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")