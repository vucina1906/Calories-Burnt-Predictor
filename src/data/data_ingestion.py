import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import logging
import json
from src.connections import s3_connection


def load_local_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"✅ Loaded local CSV: {path}, shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"❌ Error loading local file {path}: {e}")
        raise


def load_config_from_file(config_path: str = "src/connections/config.json") -> dict:
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logging.info("✅ S3 config loaded from config.json")
        return config
    except Exception as e:
        logging.error(f"❌ Failed to load config file: {e}")
        raise


def load_config_from_env() -> dict:
    try:
        config = {
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "S3_BUCKET_NAME": os.environ["S3_BUCKET_NAME"],
            "AWS_REGION": os.getenv("AWS_REGION", "us-east-1")
        }
        logging.info("✅ S3 config loaded from environment variables")
        return config
    except KeyError as e:
        logging.error(f"❌ Missing required environment variable: {e}")
        raise


def preprocess_and_merge(calories_df: pd.DataFrame, exercise_df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = exercise_df.merge(calories_df, on="User_ID")
        df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
        logging.info(f"✅ Merged dataframe shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"❌ Failed to preprocess/merge: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str = "./data") -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.info(f"✅ Saved data to {raw_data_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save data: {e}")
        raise


# ===============================
# For LOCAL use
# ===============================
""" def main_local():
    try:
        calories_path = "notebooks/calories.csv"
        exercise_path = "notebooks/exercise.csv"

        calories_df = load_local_csv(calories_path)
        exercise_df = load_local_csv(exercise_path)

        df = preprocess_and_merge(calories_df, exercise_df)

        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        save_data(train_data, test_data)

        logging.info("✅ Data ingestion (local) completed successfully.")
    except Exception as e:
        logging.error(f"❌ Data ingestion (local) failed: {e}")
        print(f"Data ingestion (local) failed: {e}") """


# ===============================
# For PRODUCTION use (S3 + ENV)
# ===============================
def main_production():
    try:
        config = load_config_from_env()

        s3 = s3_connection.s3_operations(
            bucket_name=config["S3_BUCKET_NAME"],
            aws_access_key=config["AWS_ACCESS_KEY_ID"],
            aws_secret_key=config["AWS_SECRET_ACCESS_KEY"],
            region_name=config["AWS_REGION"]
        )

        calories_df = s3.fetch_file_from_s3("calories.csv")
        exercise_df = s3.fetch_file_from_s3("exercise.csv")

        df = preprocess_and_merge(calories_df, exercise_df)

        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        save_data(train_data, test_data)

        logging.info("✅ Data ingestion (production) completed successfully.")
    except Exception as e:
        logging.error(f"❌ Data ingestion (production) failed: {e}")
        print(f"Data ingestion (production) failed: {e}")


if __name__ == "__main__":
    # Uncomment one of the following lines depending on environment:
    #main_local()        # For local development/testing
    main_production()  # For CI/CD and production with S3
