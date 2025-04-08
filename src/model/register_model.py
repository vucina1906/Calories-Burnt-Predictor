import os
import json
import mlflow
import dagshub
from src.logger import logging
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# === DAGSHUB SETUP ===

# --------- For Production (env variable auth) ---------
dagshub_token = os.getenv("CALORIES_BURNT_PRED")
if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    logging.info("✅ MLflow credentials loaded from CALORIES_BURNT_PRED")

dagshub_url = "https://dagshub.com"
repo_owner = "vucina19931906"  
repo_name = "Calories-Burnt-Predictor"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# --------- For Local Use (optional) ---------
""" mlflow.set_tracking_uri('https://dagshub.com/vucina19931906/Calories-Burnt-Predictor.mlflow')
dagshub.init(repo_owner="vucina19931906", repo_name="Calories-Burnt-Predictor", mlflow=True) """

# === FUNCTIONS ===

def load_model_info(file_path: str) -> dict:
    """Load the model info (run_id + model path) from JSON."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info(f"📁 Loaded model info from {file_path}")
        return model_info
    except Exception as e:
        logging.error(f"❌ Failed to load model info: {e}")
        raise

def register_model(model_name: str, model_info: dict):
    """Register and transition model in MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.info(f"✅ Model '{model_name}' version {model_version.version} registered and moved to STAGING.")

    except Exception as e:
        logging.error(f"❌ Model registration failed: {e}")
        raise

def main():
    try:
        model_info = load_model_info("reports/experiment_info.json")
        register_model(model_name="calories-burnt-xgb", model_info=model_info)
    except Exception as e:
        logging.error(f"🔥 Model registration process failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
