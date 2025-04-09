# scripts/promote_model.py

import os
import mlflow
from src.logger import logging

def promote_model():
    # Load DagsHub token from environment variable
    dagshub_token = os.getenv("CALORIES_BURNT_PRED")
    if not dagshub_token:
        logging.error("❌ CALORIES_BURNT_PRED environment variable is not set.")
        raise EnvironmentError("CALORIES_BURNT_PRED environment variable is not set.")

    # Set MLflow credentials
    os.environ["MLFLOW_TRACKING_USERNAME"] = "vucina19931906"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # MLflow tracking setup for DagsHub
    dagshub_url = "https://dagshub.com"
    repo_owner = "vucina19931906"
    repo_name = "Calories-Burnt-Predictor"
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    logging.info(f"📦 MLflow tracking URI set to: {dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    model_name = "calories-burnt-xgb"
    client = mlflow.MlflowClient()

    # Get the latest model version in Staging
    try:
        latest_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version
    except IndexError:
        logging.error("❌ No model found in Staging.")
        raise

    # Archive any currently deployed Production models
    for version in client.get_latest_versions(model_name, stages=["Production"]):
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )
        logging.info(f"📦 Archived old Production model version: {version.version}")

    # Promote the Staging model to Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_staging,
        stage="Production"
    )
    logging.info(f"✅ Model version {latest_staging} promoted to Production.")

if __name__ == "__main__":
    promote_model()
