import pandas as pd
from src.preprocessing.preprocess import (
    clean_raw_data, split_features_target, build_preprocessor, get_train_test_split
)


def make_sample_raw_df():
    return pd.DataFrame({
        "customerID": ["001", "002", "003", "004"],
        "gender": ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [5, 10, 20, 1],
        "PhoneService": ["Yes", "Yes", "No", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service", "No"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
        "OnlineSecurity": ["No", "Yes", "No internet service", "No"],
        "OnlineBackup": ["Yes", "No", "No internet service", "No"],
        "DeviceProtection": ["No", "Yes", "No internet service", "No"],
        "TechSupport": ["No", "No", "No internet service", "Yes"],
        "StreamingTV": ["No", "Yes", "No internet service", "No"],
        "StreamingMovies": ["No", "Yes", "No internet service", "No"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Electronic check"],
        "MonthlyCharges": [70.0, 80.5, 20.0, 60.0],
        "TotalCharges": ["350.5", " ", "400.0", "60.0"],  # includes blank string edge case
        "Churn": ["Yes", "No", "No", "Yes"],
    })


def test_clean_raw_data_converts_total_charges_to_numeric():
    df = make_sample_raw_df()
    cleaned = clean_raw_data(df)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_clean_raw_data_drops_customer_id():
    df = make_sample_raw_df()
    cleaned = clean_raw_data(df)
    assert "customerID" not in cleaned.columns


def test_split_features_target_encodes_churn_correctly():
    df = clean_raw_data(make_sample_raw_df())
    X, y = split_features_target(df)
    assert set(y.unique()).issubset({0, 1})
    assert "Churn" not in X.columns


def test_preprocessor_fits_and_transforms_without_error():
    df = clean_raw_data(make_sample_raw_df())
    X, y = split_features_target(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]


def test_get_train_test_split_returns_correct_shapes():
    df = make_sample_raw_df()
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.5, random_state=1)
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) == len(X_train)