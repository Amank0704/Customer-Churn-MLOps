# import os
# from dotenv import load_dotenv

# load_dotenv()

# # Load environment variables from .env file
# # PostgreSQL
# POSTGRES_USER = os.getenv("POSTGRES_USER")
# POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "churn_pass")
# POSTGRES_DB = os.getenv("POSTGRES_DB", "churn_db")
# POSTGRES_HOST = os.getenv("POSTGRES_HOST","localhost")
# POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# DATABASE_URL = (
#     f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
#     f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
# )


# # MLflow
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# # Paths
# RAW_DATA_PATH = "data/raw/telco_churn.csv"
# PROCESSED_DATA_PATH = "data/processed/"
# MODEL_DIR = "models/"

# TARGET_COLUMN = "Churn"





import os
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "churn_pass")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "churn_db")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

    DATABASE_URL = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

# MLflow
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

# Paths
RAW_DATA_PATH = "data/raw/telco_churn.csv"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_DIR = "models/"

TARGET_COLUMN = "Churn"