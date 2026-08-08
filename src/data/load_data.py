import pandas as pd
from src.config import RAW_DATA_PATH

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw data from a CSV file."""
    
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nTarget distribution:\n", df["Churn"].value_counts())
    