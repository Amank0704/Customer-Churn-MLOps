import time 
import joblib 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.config import MODEL_DIR
from src.data.load_data import load_raw_data
from src.preprocessing.preprocess import get_train_test_split, build_preprocessor
from src.training.evaluate import evaluate_model

# simple reasonable  default hyperparameter  
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "SVM": SVC(probability=True, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}

def build_model_pipeline(model) -> Pipeline:
    # combining preprocessing and model in one single pipeline so that training and interface always use identical transforms
    
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    return pipeline


def train_all_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    # Train every model in MODELS, evaluate it, and return a results DataFrame.
    results = []

    for name, model in MODELS.items():
        print(f"Training {name}...")
        start = time.time()

        pipeline = build_model_pipeline(model)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        duration = round(time.time() - start, 2)

        results.append({
            "model_name": name,
            "pipeline": pipeline,
            "train_time_sec": duration,
            **metrics
        })
        print(f"  Done in {duration}s | ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")

    return pd.DataFrame(results)


def select_best_model(results_df: pd.DataFrame):
    """Pick the best model primarily by ROC-AUC, tie-broken by F1."""
    sorted_df = results_df.sort_values(
        by=["roc_auc", "f1"], ascending=[False, False]
    )
    best_row = sorted_df.iloc[0]
    return best_row


if __name__ == "__main__":
    df = load_raw_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    results_df = train_all_models(X_train, y_train, X_test, y_test)

    print("\n=== Model Comparison ===")
    print(results_df[["model_name", "accuracy", "precision", "recall", "f1", "roc_auc", "train_time_sec"]]
          .sort_values(by="roc_auc", ascending=False)
          .to_string(index=False))

    best = select_best_model(results_df)
    print(f"\nBest model: {best['model_name']} (ROC-AUC: {best['roc_auc']:.4f}, F1: {best['f1']:.4f})")

    joblib.dump(best["pipeline"], f"{MODEL_DIR}best_model.pkl")
    print(f"Saved best model to {MODEL_DIR}best_model.pkl")