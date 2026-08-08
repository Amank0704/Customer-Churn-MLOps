import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from src.config import TARGET_COLUMN

# column that are identifier or leak no useful signal
DROP_COLUMNS = ["customerID"]

NUMERIC_FEATURES = ["tenure","MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    #Total charge
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = "coerce")
    
    #Drop exact duplicate
    df = df.drop_duplicates()
    
    
    #Drop identifier
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    
    return df


def split_features_target(df: pd.DataFrame):
    """ seperate features X from target y 
    encode target to 0/1
    """
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    return X, y

def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing pipeline.
    This is fit ONLY on training data to prevent data leakage.
    """
    
    
    
    # Numeric: impute missing with median, then scale
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical: impute missing with most frequent, then one-hot encode.
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        
    ])
    return preprocessor

def get_train_test_split(df: pd.DataFrame,test_size:float=0.2,random_state: int=42):
    # Clean data , then split into features and target then train and test 
    
    df_clean = clean_raw_data(df)
    X, y = split_features_target(df_clean)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = random_state, stratify=y
        
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from src.data.load_data import load_raw_data
    
    df = load_raw_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Train churn rate:", y_train.mean().round(3))
    print("Test churn rate:", y_test.mean().round(3))