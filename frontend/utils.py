import os
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

def call_predict(payload: dict) -> dict:
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def call_model_info() -> dict:
    response = requests.get(f"{API_URL}/model-info", timeout=10)
    response.raise_for_status()
    return response.json()