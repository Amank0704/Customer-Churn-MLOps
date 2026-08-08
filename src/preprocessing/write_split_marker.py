# this file is just to prove that preprocessing stage ran

from src.data.load_data import load_raw_data
from src.preprocessing.preprocess import get_train_test_split

if __name__ == "__main__":
    df = load_raw_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    with open("data/processed/train_test_split_marker.txt", "w") as f:
        f.write(f"train_rows={X_train.shape[0]}\n")
        f.write(f"test_rows={X_test.shape[0]}\n")
        f.write(f"train_churn_rate={y_train.mean():.4f}\n")