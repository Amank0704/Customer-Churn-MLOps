import pandas as pd

#column we expect in IBM telco churn dataset
# Kept as a flat list so it's easy to read and explain in an interview.

EXPECTED_COLUMNS = [
     "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]


class DataValidationError(Exception):
    """ Raised when the input dataset does not match the expected schema."""
    pass

def validate_schema(df: pd.DataFrame) -> None:
    """ Check that all expected columns are present"""
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing expected columns: {missing_cols}")
    

def validate_target(df: pd.DataFrame) -> None:
    allowed = {"Yes", "No"}
    actual = set(df["Churn"].dropna().unique())
    if not actual.issubset(allowed):
        raise DataValidationError(f"Unexpected value in churn column: {actual}")
    
def validate_not_empty(df: pd.DataFrame) -> None:
    """ Check that dataset is not empty"""
    if df.shape[0] == 0:
        raise DataValidationError("Dataset is empty")
        

def run_all_validations(df: pd.DataFrame) -> None:
    """Run every validation check. Raises DataValidationError on first failure."""
    validate_not_empty(df)
    validate_schema(df)
    validate_target(df)
    print("All validations passed.")


if __name__ == "__main__":
    from src.data.load_data import load_raw_data

    df = load_raw_data()
    run_all_validations(df)   
    
    