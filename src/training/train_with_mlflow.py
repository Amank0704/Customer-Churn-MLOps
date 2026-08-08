# import mlflow
# import mlflow.sklearn
# import joblib


# from src.config import MLFLOW_TRACKING_URI, MODEL_DIR
# from src.data.load_data import load_raw_data
# from src.preprocessing.preprocess import get_train_test_split
# from src.training.train import build_model_pipeline, MODELS, select_best_model
# from src.training.evaluate import evaluate_model
# from src.training.save_model_info import save_model_info


# EXPERIMENT_NAME = "customer_churn"
# REGISTERED_MODEL_NAME = "churn_best_model"

# def log_model_run(name, pipeline, metrics, params):
#     # log one model params,metrics, and artifacts to MlFlow
    
#     with mlflow.start_run(run_name = name):
#         mlflow.log_param("model_name", name)
#         for key, value in params.items():
#             mlflow.log_param(key, value)
            
#         for metric_name, metric_value in metrics.items():
#             mlflow.log_metric(metric_name, metric_value)
            
#         mlflow.sklearn.log_model(pipeline, artifact_path="model")
        
#         return mlflow.active_run().info.run_id
    
    
# def train_and_log_all():
#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#     mlflow.set_experiment(EXPERIMENT_NAME)

#     df = load_raw_data()
#     X_train, X_test, y_train, y_test = get_train_test_split(df)

#     results = []

#     for name, model in MODELS.items():
#         print(f"Training + logging {name}...")
#         pipeline = build_model_pipeline(model)
#         pipeline.fit(X_train, y_train)

#         metrics = evaluate_model(pipeline, X_test, y_test)
#         params = model.get_params()
#         # Keep only simple, loggable params (skip nested objects)
#         simple_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool)) or v is None}

#         run_id = log_model_run(name, pipeline, metrics, simple_params)

#         results.append({
#             "model_name": name,
#             "pipeline": pipeline,
#             "run_id": run_id,
#             **metrics
#         })

#     import pandas as pd
#     results_df = pd.DataFrame(results)
#     best = select_best_model(results_df)

#     print(f"\nBest model: {best['model_name']} (ROC-AUC: {best['roc_auc']:.4f})")

#     # Register the best model in MLflow Model Registry
#     model_uri = f"runs:/{best['run_id']}/model"
#     registered = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
#     print(f"Registered model '{REGISTERED_MODEL_NAME}' version {registered.version}")

#     # Also save locally for the API to load without needing MLflow at inference time
#     joblib.dump(best["pipeline"], f"{MODEL_DIR}best_model.pkl")

#     from src.training.save_model_info import save_model_info
#     save_model_info(best["model_name"], best, registered.version)
#     print("Saved model_info.json")
    
#     return results_df, best


# if __name__ == "__main__":
#     train_and_log_all()



























# v2


import mlflow
import mlflow.sklearn
import joblib


from src.config import MLFLOW_TRACKING_URI, MODEL_DIR
from src.data.load_data import load_raw_data
from src.preprocessing.preprocess import get_train_test_split
from src.training.train import build_model_pipeline, MODELS, select_best_model
from src.training.evaluate import evaluate_model
from src.training.save_model_info import save_model_info

EXPERIMENT_NAME = "customer_churn"
REGISTERED_MODEL_NAME = "churn_best_model"

def log_model_run(name, pipeline, metrics, params):
    # log one model params,metrics, and artifacts to MlFlow
    
    with mlflow.start_run(run_name = name):
        mlflow.log_param("model_name", name)
        for key, value in params.items():
            mlflow.log_param(key, value)
            
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        
        return mlflow.active_run().info.run_id
    
    
def train_and_log_all():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    results = []

    for name, model in MODELS.items():
        print(f"Training + logging {name}...")
        pipeline = build_model_pipeline(model)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        params = model.get_params()
        # Keep only simple, loggable params (skip nested objects)
        simple_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool)) or v is None}

        run_id = log_model_run(name, pipeline, metrics, simple_params)

        results.append({
            "model_name": name,
            "pipeline": pipeline,
            "run_id": run_id,
            **metrics
        })

    import pandas as pd
    results_df = pd.DataFrame(results)
    best = select_best_model(results_df)

    print(f"\nBest model: {best['model_name']} (ROC-AUC: {best['roc_auc']:.4f})")

    # Register the best model in MLflow Model Registry
    model_uri = f"runs:/{best['run_id']}/model"
    
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    print(
        f"Registered model '{REGISTERED_MODEL_NAME}' "
        f"version {registered.version}"
    )

    joblib.dump(
        best["pipeline"],
        f"{MODEL_DIR}best_model.pkl"
    )

    save_model_info(
        model_name=best["model_name"],
        metrics=best,
        version=registered.version,
        comparison=results_df
    )
   
   
    print("Saved model_info.json")
    
    return results_df, best


if __name__ == "__main__":
    train_and_log_all()
