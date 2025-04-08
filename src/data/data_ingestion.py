import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import logging
import json
from src.connections import s3_connection

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file from local path."""
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        raise
def load_s3_config(config_path: str = "src/connections/config.json") -> dict:
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logging.info("✅ S3 config loaded from config.json")
        return config
    except Exception as e:
        logging.error(f"❌ Failed to load S3 config: {e}")
        raise

def load_csv_from_s3(bucket: str, filename: str, access_key: str, secret_key: str) -> pd.DataFrame:
    """ Load a CSV file from S3. """
    try:
        s3 = s3_connection.s3_operations(bucket, access_key, secret_key)
        df = s3.fetch_file_from_s3(filename)
        logging.info(f"Loaded {filename} from S3 bucket {bucket} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch {filename} from S3: {e}")
        raise 

def preprocess_and_merge(calories_df: pd.DataFrame, exercise_df: pd.DataFrame) -> pd.DataFrame:
    """Merge and preprocess the calories and exercise data."""
    try:
        df = exercise_df.merge(calories_df, on="User_ID")
        df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
        logging.info(f"Merged data shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error during merging/preprocessing: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug(f"Train and test data saved to {raw_data_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving the data: {e}")
        raise

def main():
    try:
        # === Local file paths ===
        #calories_path = "notebooks/calories.csv"
        #exercise_path = "notebooks/exercise.csv"

        # === Loading from S3 bucket using AWS credentials stored in config.json file ===
        config = load_s3_config()
        bucket = config["s3_bucket"]
        access_key = config["aws_access_key"]
        secret_key = config["aws_secret_key"]
        region = config.get("region", "us-east-1")

        s3 = s3_connection.s3_operations(bucket, access_key, secret_key, region)
        calories_df = s3.fetch_file_from_s3("calories.csv")
        exercise_df = s3.fetch_file_from_s3("exercise.csv")

        df = preprocess_and_merge(calories_df, exercise_df)

        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

        save_data(train_data, test_data, data_path="./data")

        logging.info("Data ingestion completed successfully.")

    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        print(f"Data ingestion failed: {e}")

if __name__ == "__main__":
    main()
