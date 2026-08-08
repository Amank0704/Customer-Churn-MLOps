


import os
import requests


API_URL = os.getenv("API_URL")

if not API_URL:
    raise RuntimeError(
        "API_URL is not configured. "
        "Please configure the deployed FastAPI backend URL."
    )

API_URL = API_URL.strip().rstrip("/")


def call_predict(payload: dict) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def call_model_info() -> dict:
    response = requests.get(
        f"{API_URL}/model-info",
        timeout=30
    )

    response.raise_for_status()

    return response.json()