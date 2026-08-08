# Customer Churn Prediction — End-to-End MLOps System

<p align="center">
  <a href="https://fastapi.tiangolo.com/"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white"></a>
  <a href="https://mlflow.org/"><img alt="MLflow" src="https://img.shields.io/badge/MLflow-2.x-0194E2?logo=mlflow&logoColor=white"></a>
  <a href="https://dvc.org/"><img alt="DVC" src="https://img.shields.io/badge/DVC-3.x-945DD6?logo=dvc&logoColor=white"></a>
  <a href="https://streamlit.io/"><img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white"></a>
  <a href="https://www.postgresql.org/"><img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white"></a>
  <a href="https://docs.evidentlyai.com/"><img alt="Evidently" src="https://img.shields.io/badge/Evidently-drift_monitoring-6C63FF"></a>
  <a href="https://www.docker.com/"><img alt="Docker" src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-brightgreen"></a>
</p>

<p align="center"><b>Multi-model churn prediction with MLflow tracking · FastAPI serving · Evidently drift monitoring</b></p>

<p align="center">
  <a href="#live-demo">Live Demo</a> ·
  <a href="#tech-stack">Tech Stack</a> ·
  <a href="#local-setup">Quick Start</a> ·
  <a href="#api-documentation">API Reference</a> ·
  <a href="#docker-setup">Deployment</a> ·
  <a href="#monitoring--drift-detection">Monitoring</a>
</p>

---

A production-style, end-to-end MLOps system for predicting customer churn, built to demonstrate the **full ML lifecycle** — from raw data to a monitored, retrainable service — using a simple, interview-defensible tech stack.

## Live Demo

- **Dashboard:** [https://customer-churn-dashboard-production-ec7a.up.railway.app/](Live_DASHBOARD)
- **Backend API:** [https://customer-churn-mlops-production-e86d.up.railway.app/](BACKEND_API_URL) (interactive docs at `BACKEND_API_URL/docs`)

> **Note:** The system also runs fully **locally / via Docker Compose** — see [Local Setup](#local-setup) and [Docker Setup](#docker-setup) below.

---

## Table of Contents

- [Live Demo](#live-demo)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Training & Results](#model-training--results)
- [API Documentation](#api-documentation)
- [Local Setup](#local-setup)
- [Docker Setup](#docker-setup)
- [DVC Pipeline](#dvc-pipeline)
- [Monitoring & Drift Detection](#monitoring--drift-detection)
- [Retraining Workflow](#retraining-workflow)
- [Testing](#testing)
- [AWS Deployment (Planned)](#aws-deployment-planned)

---

## Problem Statement

Telecom companies lose significant revenue to customer churn. This project builds a system that:

1. Predicts whether a customer is likely to churn, using account and service usage data.
2. Tracks and compares multiple models systematically (not just one notebook run).
3. Serves predictions through a real API with request validation and logging.
4. Monitors incoming data for drift over time.
5. Supports manual retraining with automatic promotion only if the new model outperforms production.

The goal was **operational realism at fresher-appropriate complexity** — every MLOps concept here is implemented for real, not simulated, but without unnecessary infrastructure (no Kubernetes, no Kafka, no Airflow).

---

## Architecture

```
User
  │
  ▼
Streamlit  (dashboard: predict, compare, model info, monitoring)
  │
  ▼
FastAPI    (inference API)
  │
  ▼
Production Model (loaded from MLflow Registry / local artifact)
  │
  ▼
PostgreSQL (prediction logs)


Training (offline):
Dataset → DVC → Preprocessing → Train 7 Models → Evaluate
   → MLflow Tracking → Best Model → Model Registry → Production


Monitoring:
Production Inputs (from logs) → Evidently → Drift Detection → Streamlit Monitoring Page


Deployment:
GitHub → GitHub Actions (test + build) → Docker Compose (local/EC2)
```

**Key design decision:** Training and serving are fully decoupled. The API never retrains on startup — it only loads an already-trained, already-validated artifact. This mirrors how real ML systems separate the training pipeline from the serving path.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data & ML | Pandas, NumPy, Scikit-learn |
| Experiment Tracking | MLflow (tracking + model registry) |
| Data/Pipeline Versioning | DVC |
| API | FastAPI + Pydantic |
| Database | PostgreSQL + SQLAlchemy |
| Frontend | Streamlit |
| Monitoring | Evidently (data drift detection) |
| Testing | Pytest |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Cloud (planned) | AWS EC2, RDS, S3 |

---

## Project Structure

```
customer-churn-mlops/
├── api/                    # FastAPI service
│   ├── main.py
│   ├── schemas.py
│   └── model_loader.py
├── src/
│   ├── data/                # Loading + schema validation
│   ├── preprocessing/       # Cleaning + sklearn Pipeline
│   ├── training/             # Training, evaluation, MLflow, retraining
│   └── database/             # SQLAlchemy models + session
├── frontend/                 # Streamlit multipage dashboard
│   └── pages/
├── monitoring/                # Evidently drift checks
├── tests/                      # Pytest suite
├── data/raw/                    # Dataset (DVC-tracked, not committed)
├── models/                      # Trained model artifacts (DVC-tracked)
├── .github/workflows/            # CI + deploy pipelines
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
└── .env.example
```

---

## Dataset

**IBM Telco Customer Churn** dataset — ~7,043 customer records with 20 features (demographics, account info, service usage) and a binary `Churn` (Yes/No) target.

- Source: Kaggle — search "Telco Customer Churn" (`blastchar/telco-customer-churn`)
- Place the file at: `data/raw/telco_churn.csv`
- Tracked via DVC (see [DVC Pipeline](#dvc-pipeline)) — not committed to Git directly

**Known data quirk handled explicitly:** `TotalCharges` is stored as a string with blank values for customers with zero tenure — this is converted to numeric with `errors="coerce"` before training, and the same cleaning logic is reused for the monitoring reference data.

---

## Model Training & Results

Seven classification models are trained and compared on identical preprocessing:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors
5. Support Vector Machine
6. Gradient Boosting
7. AdaBoost

**Why ROC-AUC and F1 as primary metrics (not accuracy):** The dataset is class-imbalanced (~26% churn), making accuracy alone a misleading evaluation metric. Therefore, ROC-AUC and F1-score were used as the primary metrics. ROC-AUC measures the model's ability to distinguish churners from non-churners across all classification thresholds, while F1-score balances precision and recall, minimizing both missed churners (false negatives) and unnecessary retention efforts (false positives). The best-performing Gradient Boosting model achieved a ROC-AUC of 0.8434, an F1-score of 0.5895, and an accuracy of 80.62%, demonstrating strong ranking performance and overall classification accuracy while maintaining a balanced trade-off between identifying churners and limiting false alarms.

All models are trained inside a single `Pipeline` combining preprocessing (`ColumnTransformer`) and the classifier — the *same object* is saved and used at inference time, which prevents any mismatch between how features were transformed during training vs. serving.


| Model                   |  Accuracy | Precision |    Recall |  F1-Score |   ROC-AUC |
| ----------------------- | --------: | --------: | --------: | --------: | --------: |
| **Gradient Boosting** ⭐ | **0.806** | **0.674** |     0.524 |     0.589 | **0.843** |
| Logistic Regression     |     0.806 |     0.657 | **0.559** | **0.604** |     0.842 |
| Random Forest           |     0.801 |     0.660 |     0.513 |     0.577 |     0.840 |
| AdaBoost                |     0.798 |     0.646 |     0.527 |     0.580 |     0.840 |
| K-Nearest Neighbors     |     0.782 |     0.595 |     0.559 |     0.577 |     0.822 |
| Decision Tree           |     0.776 |     0.591 |     0.511 |     0.548 |     0.801 |
| Support Vector Machine  |     0.791 |     0.639 |     0.487 |     0.552 |     0.793 |


The best model (by ROC-AUC, tie-broken by F1) is registered in the **MLflow Model Registry** and saved locally as `models/best_model.pkl` for the API to load.

---

## API Documentation

Base URL (deployed): `PLACEHOLDER_API_URL`
Base URL (local): `http://localhost:8000`

### `GET /health`
Returns service liveness status.
```json
{"status": "ok"}
```

### `GET /model-info`
Returns metadata about the current production model.
```json
{
  "model_name": "GradientBoosting",
  "model_version": "1",
  "roc_auc": 0.845,
  "f1": 0.578,
  "accuracy": 0.804,
  "training_date": "2026-08-07 10:30:00"
}
```

### `POST /predict`
Accepts a single customer's features (validated via Pydantic) and returns a prediction.

**Request body:** see `api/schemas.py::CustomerInput` for the full schema (all 19 features, e.g. `tenure`, `Contract`, `MonthlyCharges`, etc.)

**Response:**
```json
{
  "prediction": "Churn",
  "probability": 0.82,
  "model_version": "1"
}
```

Every successful prediction is logged to PostgreSQL (`prediction_logs` table) with timestamp, model version, input features, prediction, and probability — this log is what powers the Monitoring dashboard and drift detection.

Interactive Swagger docs available at `/docs` on both the deployed API (`PLACEHOLDER_API_URL/docs`) and locally (`http://localhost:8000/docs`).

---

## Local Setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd customer-churn-mlops

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your local PostgreSQL credentials

# 5. Place the dataset
# Download from Kaggle and save as: data/raw/telco_churn.csv

# 6. Validate the dataset
python -m src.data.validate_data

# 7. Start MLflow server (separate terminal)
mlflow server --host 0.0.0.0 --port 5000

# 8. Train models (logs to MLflow, registers best model)
python -m src.training.train_with_mlflow

# 9. Set up PostgreSQL and create tables
python -m src.database.db

# 10. Run the API
uvicorn api.main:app --reload --port 8000

# 11. Run the dashboard (separate terminal)
streamlit run frontend/app.py
```

Visit:
- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`

---

## Docker Setup

Runs all four services (PostgreSQL, MLflow, FastAPI, Streamlit) together:

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`

To stop:
```bash
docker compose down
```

> Note: models must be trained at least once (locally, before containerizing, or by running the training script inside the `api` container) so `models/best_model.pkl` exists for the API to load.

---

## DVC Pipeline

Dataset and pipeline versioning via DVC:

```bash
dvc init
dvc add data/raw/telco_churn.csv
dvc repro          # Runs: validate → preprocess → train → evaluate
dvc dag             # Visualize the pipeline graph
dvc push / dvc pull # Sync artifacts with remote storage
```

`dvc repro` only re-runs stages whose dependencies actually changed, making the pipeline reproducible without re-running everything on every change.

---

## Monitoring & Drift Detection

The Streamlit **Monitoring** page shows:

- Total prediction count
- Churn vs. Stay distribution
- Average prediction confidence
- Data drift status (via Evidently)

Drift is computed by comparing the original training data distribution against recent production inputs (pulled from the `prediction_logs` table). If fewer than 30 production predictions have been logged, the dashboard explicitly shows:

> *"Insufficient production data for drift analysis."*

instead of a misleading or fabricated result — statistical drift tests are unreliable on very small samples.

---

## Retraining Workflow

Manually triggered:

```bash
python -m src.training.retrain
```

Flow: `New Data → Validate → Preprocess → Train 7 Models → Compare → Best Candidate → Compare with Current Production (by ROC-AUC) → Promote only if Better`

This reuses the same MLflow training pipeline as initial training — retraining is not a separate code path, just a different trigger followed by a promotion check. No automated scheduling is implemented; this is intentionally manual for this project's scope.

---

## Testing

```bash
python -m pytest tests/ -v
```

Covers:
- Schema and target validation (`test_data.py`)
- Data cleaning and preprocessing pipeline correctness (`test_preprocessing.py`)
- API health check, valid/invalid prediction requests (`test_api.py`, using mocked model loading)

---

## AWS Deployment (Planned)

The project is architected to deploy to AWS with minimal changes:

- **EC2** — hosts FastAPI, Streamlit, and MLflow via the same `docker-compose.yml`
- **RDS (PostgreSQL)** — replaces the local Postgres container; only `.env` changes
- **S3** — replaces local MLflow artifact storage; only the MLflow service's `--default-artifact-root` changes
- **GitHub Actions** — `deploy.yml` is already scaffolded with the SSH-to-EC2 deploy step commented out, ready to enable once EC2/RDS/S3 are provisioned

This separation (environment variables driving all connections) is what makes the cloud migration a configuration change rather than a code change.
