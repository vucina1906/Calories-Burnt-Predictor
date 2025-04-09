import os
import pandas as pd
import mlflow
import unittest
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TestCaloriesModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up MLflow DagsHub credentials for production use
        dagshub_token = os.getenv("CALORIES_BURNT_PRED")
        if not dagshub_token:
            raise EnvironmentError("CALORIES_BURNT_PRED environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = "vucina19931906"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "vucina19931906"
        repo_name = "Calories-Burnt-Predictor"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        cls.model_name = "calories-burnt-xgb"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # Load holdout features and target data
        cls.X_holdout = pd.read_csv("data/processed/test_features.csv")
        cls.y_holdout = pd.read_csv("data/processed/test_target.csv").squeeze()

    @staticmethod
    def get_latest_model_version(model_name, stage="Production"):
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise ValueError(f"No model found in stage: {stage}")
        return versions[0].version

    def test_model_is_loaded(self):
        """Test that model is loaded from MLflow."""
        self.assertIsNotNone(self.model, "Model failed to load from MLflow.")

    def test_model_signature(self):
        """Test input-output compatibility of the model."""
        input_df = self.X_holdout.head(1)
        prediction = self.model.predict(input_df)
        self.assertEqual(len(prediction), 1)
        self.assertTrue(np.isscalar(prediction[0]) or isinstance(prediction[0], (float, int)))

    def test_model_performance_threshold(self):
        """Ensure model meets minimum performance thresholds."""
        y_pred = self.model.predict(self.X_holdout)
        mae = mean_absolute_error(self.y_holdout, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_holdout, y_pred))
        r2 = r2_score(self.y_holdout, y_pred)

        # Set acceptable threshold values
        self.assertLessEqual(mae, 100, "MAE is too high")
        self.assertLessEqual(rmse, 150, "RMSE is too high")
        self.assertGreaterEqual(r2, 0.80, "R² is too low")


if __name__ == "__main__":
    unittest.main()
