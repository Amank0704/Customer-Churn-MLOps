import pandas as pd
import pytest

from src.data.validate_data import (
    validate_schema, validate_target, validate_not_empty,
    run_all_validations, DataValidationError, EXPECTED_COLUMNS
)


def make_valid_df():
    """Build a minimal valid DataFrame matching the expected schema."""
    data = {col: ["sample"] for col in EXPECTED_COLUMNS}
    data["Churn"] = ["Yes"]
    return pd.DataFrame(data)


def test_validate_schema_passes_with_all_columns():
    df = make_valid_df()
    validate_schema(df)  # should not raise


def test_validate_schema_fails_with_missing_column():
    df = make_valid_df().drop(columns=["Churn"])
    with pytest.raises(DataValidationError):
        validate_schema(df)


def test_validate_target_passes_with_yes_no():
    df = make_valid_df()
    df["Churn"] = ["Yes"]
    validate_target(df)  # should not raise


def test_validate_target_fails_with_invalid_value():
    df = make_valid_df()
    df["Churn"] = ["Maybe"]
    with pytest.raises(DataValidationError):
        validate_target(df)


def test_validate_not_empty_fails_on_empty_df():
    df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    with pytest.raises(DataValidationError):
        validate_not_empty(df)


def test_run_all_validations_passes_on_valid_data():
    df = make_valid_df()
    run_all_validations(df)  # should not raise