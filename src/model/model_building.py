import numpy as np
import pandas as pd
import pickle
import os
from xgboost import XGBRegressor
from src.logger import logging

def load_data(features_path: str, target_path: str) -> tuple:
    """Load features and target from CSV files."""
    try:
        X = pd.read_csv(features_path)
        y = pd.read_csv(target_path).squeeze()  # to Series
        logging.info(f"Loaded features from {features_path}, shape: {X.shape}")
        logging.info(f"Loaded target from {target_path}, shape: {y.shape}")
        return X.values, y.values
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    """Train XGBoost regressor with best hyperparameters that we got in testing phase"""
    try:
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            gamma=0.3,
            subsample=0.6,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        logging.info("✅ XGBoost model trained with tuned hyperparameters.")
        return model
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"✅ Model saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main():
    try:
        X_train, y_train = load_data(
            "./data/processed/train_features.csv",
            "./data/processed/train_target.csv"
        )
        model = train_model(X_train, y_train)
        save_model(model, "models/model.pkl")
    except Exception as e:
        logging.error(f"Model building failed: {e}")
        print(f"Model building failed: {e}")

if __name__ == "__main__":
    main()
