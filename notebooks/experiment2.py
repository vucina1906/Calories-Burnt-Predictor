import os
import time
import logging
import mlflow
import dagshub
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ========================== CONFIG ==========================
CONFIG = {
    "calories_path": "calories.csv",
    "exercise_path": "exercise.csv",
    "test_size": 0.2,
    "random_state": 42,
    "experiment_name": "Model Comparasion",
    "mlflow_tracking_uri": "https://dagshub.com/vucina19931906/Calories-Burnt-Predictor.mlflow",
    "dagshub_repo_owner": "vucina19931906",
    "dagshub_repo_name": "Calories-Burnt-Predictor"
}


# ========================== MODELS ==========================
MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=CONFIG["random_state"]),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=CONFIG["random_state"]),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=CONFIG["random_state"], objective='reg:squarederror')
}

# ========================== LOGGING ==========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========================== DAGSHUB SETUP ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== DATA LOADING ==========================
def load_data(calories_path="calories.csv", exercise_path="exercise.csv"):
    try:
        logging.info("Loading calories.csv and exercise.csv...")
        calories = pd.read_csv(calories_path)
        exercise_data = pd.read_csv(exercise_path)

        logging.info("Merging datasets on 'User_ID'...")
        df = exercise_data.merge(calories, on="User_ID")

        logging.info("Encoding 'Gender' column...")
        df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})

        logging.info("Splitting features and target...")
        X = df.drop(columns=['User_ID', 'Calories'])
        y = df['Calories']

        return train_test_split(X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])
    except Exception as e:
        logging.error(f"Error in load_data: {e}")
        raise


# ========================== METRIC LOGGING ==========================
def log_metrics(y_test, preds):
    mlflow.log_metrics({
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2_score": r2_score(y_test, preds)
    })

# ========================== PARAM LOGGING ==========================
def log_model_params(name, model):
    params = {}
    if name == "RandomForest":
        params["n_estimators"] = model.n_estimators
        params["max_depth"] = model.max_depth
    elif name == "GradientBoosting":
        params["n_estimators"] = model.n_estimators
        params["learning_rate"] = model.learning_rate
        params["max_depth"] = model.max_depth
    elif name == "XGBoost":
        params.update(model.get_params())

    mlflow.log_params(params)

# ========================== TRAIN & EVALUATE ==========================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="Regression Models") as parent_run:
        for model_name, model in MODELS.items():
            with mlflow.start_run(run_name=model_name, nested=True):
                try:
                    logging.info(f"Training {model_name}...")
                    model.fit(X_train, y_train)

                    logging.info("Logging model parameters...")
                    mlflow.log_param("model", model_name)
                    log_model_params(model_name, model)

                    logging.info("Predicting...")
                    preds = model.predict(X_test)

                    logging.info("Logging evaluation metrics...")
                    log_metrics(y_test, preds)

                    logging.info("Logging model artifact...")
                    mlflow.sklearn.log_model(model, "model")

                    logging.info(f"{model_name} evaluation completed.\n")

                except Exception as e:
                    logging.error(f"Error with {model_name}: {e}")
                    mlflow.log_param("error", str(e))

# ========================== MAIN ==========================
if __name__ == "__main__":
    logging.info("Loading data...")
    X_train, X_test, y_train, y_test = load_data(
    calories_path=CONFIG["calories_path"],
    exercise_path=CONFIG["exercise_path"])
    train_and_evaluate(X_train, X_test, y_train, y_test)
