import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from src.logger import logging
import dagshub

# =======================
# For production use (GitHub Actions / CI/CD)
# =======================
dagshub_token = os.getenv("CALORIES_BURNT_PRED")
dagshub_username = "vucina19931906"  # ✅ Set your correct DagsHub username here

if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    logging.info("✅ MLflow credentials loaded from GitHub Secrets (CALORIES_BURNT_PRED)")
else:
    logging.error("❌ CALORIES_BURNT_PRED not found in environment variables")

dagshub_url = "https://dagshub.com"
repo_owner = dagshub_username
repo_name = "Calories-Burnt-Predictor"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Calories-Burnt-Evaluation")

# =======================
# For local use (Uncomment if testing locally)
# =======================
""" mlflow.set_tracking_uri('https://dagshub.com/vucina19931906/Calories-Burnt-Predictor.mlflow')
dagshub.init(repo_owner="vucina19931906", repo_name="Calories-Burnt-Predictor", mlflow=True)
mlflow.set_experiment("Calories-Burnt-Evaluation") """


def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def load_data(features_path: str, target_path: str) -> tuple:
    try:
        X = pd.read_csv(features_path)
        y = pd.read_csv(target_path).squeeze()
        logging.info(f"Loaded test features from {features_path} and target from {target_path}")
        return X.values, y.values
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred)
        }
        logging.info(f"Evaluation metrics calculated: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise

def save_metrics(metrics: dict, filepath: str):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info(f"Saved metrics to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")
        raise

def save_model_info(run_id: str, model_path: str, filepath: str):
    try:
        info = {"run_id": run_id, "model_path": model_path}
        with open(filepath, 'w') as file:
            json.dump(info, file, indent=4)
        logging.debug(f"Model info saved to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save model info: {e}")
        raise

def main():
    try:
        logging.info("📦 Starting model evaluation stage...")
        logging.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        with mlflow.start_run() as run:
            model = load_model("models/model.pkl")
            X_test, y_test = load_data("data/processed/test_features.csv", "data/processed/test_target.csv")
            
            metrics = evaluate_model(model, X_test, y_test)
            save_metrics(metrics, "reports/metrics.json")

            # Log metrics to MLflow
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log model parameters
            if hasattr(model, "get_params"):
                for param, val in model.get_params().items():
                    mlflow.log_param(param, val)

            # Log model & artifacts
            mlflow.sklearn.log_model(model, "model")
            save_model_info(run.info.run_id, "model", "reports/experiment_info.json")
            mlflow.log_artifact("reports/metrics.json")

            logging.info("✅ Model evaluation and logging complete.")
    
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
