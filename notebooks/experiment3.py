import os
import logging
import time
import mlflow
import dagshub
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========== CONFIG ==========
CONFIG = {
    "calories_path": "calories.csv",
    "exercise_path": "exercise.csv",
    "test_size": 0.2,
    "random_state": 42,
    "experiment_name": "XGBoost Hyperparameter Tuning",
    "mlflow_tracking_uri": "https://dagshub.com/vucina19931906/Calories-Burnt-Predictor.mlflow",  # replace
    "dagshub_repo_owner": "vucina19931906",  # replace
    "dagshub_repo_name": "Calories-Burnt-Predictor"
}

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== DAGSHUB & MLFLOW SETUP ==========
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========== LOAD AND PREPARE DATA ==========
def load_data(calories_path, exercise_path):
    logging.info("Loading and merging datasets...")
    calories = pd.read_csv(calories_path)
    exercise = pd.read_csv(exercise_path)
    df = exercise.merge(calories, on="User_ID")
    df["Gender"] = df["Gender"].map({"male": 0, "female": 1})
    X = df.drop(columns=["User_ID", "Calories"])
    y = df["Calories"]
    return train_test_split(X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])

# ========== PARAM GRID ==========
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}

grid = list(ParameterGrid(param_grid))

# ========== TRAIN AND LOG TUNED MODELS ==========
def run_grid_search(X_train, y_train, X_test, y_test):
    logging.info(f"Total combinations to evaluate: {len(grid)}")
    kf = KFold(n_splits=5, shuffle=True, random_state=CONFIG["random_state"])

    results = []

    with mlflow.start_run(run_name="XGBoost Tuning") as parent_run:
        for params in tqdm(grid, desc="GridSearch XGBoost"):
            with mlflow.start_run(run_name=f"params={params}", nested=True):
                try:
                    model = XGBRegressor(**params, objective='reg:squarederror', random_state=42, n_jobs=-1)

                    scores = []
                    for train_idx, val_idx in kf.split(X_train):
                        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                        model.fit(X_tr, y_tr)
                        preds = model.predict(X_val)
                        score = mean_absolute_error(y_val, preds)
                        scores.append(score)

                    avg_score = np.mean(scores)

                    mlflow.log_params(params)
                    mlflow.log_metric("cv_mae", avg_score)
                    results.append((params, avg_score))

                except Exception as e:
                    logging.error(f"Error in combination {params}: {e}")
                    mlflow.log_param("error", str(e))

    return sorted(results, key=lambda x: x[1])

# ========== MAIN ==========
if __name__ == "__main__":
    start = time.time()
    X_train, X_test, y_train, y_test = load_data(CONFIG["calories_path"], CONFIG["exercise_path"])

    # Reset index for KFold split compatibility
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    sorted_results = run_grid_search(X_train, y_train, X_test, y_test)
    best_params, best_mae = sorted_results[0]

    logging.info(f"\nBest parameters: {best_params}")
    logging.info(f"Best CV MAE: {best_mae:.4f}")

    # Final model training and logging
    best_model = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    with mlflow.start_run(run_name="Best Model Final Evaluation"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({"final_mae": mae, "final_rmse": rmse, "final_r2": r2})
        mlflow.sklearn.log_model(best_model, "final_model")
        logging.info(f"Final evaluation logged: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        logging.info("Model saved and logged successfully.")

    logging.info(f"Total tuning time: {time.time() - start:.2f} seconds.")
