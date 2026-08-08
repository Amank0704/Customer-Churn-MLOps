import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


VALID_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 350.5
}


def test_predict_with_invalid_input_returns_422():
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["Contract"] = "Invalid Value"  # not in allowed Literal set
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_with_missing_field_returns_422():
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["tenure"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


@patch("api.main.save_prediction_log")
@patch("api.main.load_model_info")
@patch("api.main.load_model")
def test_predict_with_valid_input_returns_200(mock_load_model, mock_load_info, mock_save_log):
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.3, 0.7]]
    mock_load_model.return_value = mock_model
    mock_load_info.return_value = {"model_version": "1"}

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Churn"
    assert data["probability"] == 0.7
    assert data["model_version"] == "1"