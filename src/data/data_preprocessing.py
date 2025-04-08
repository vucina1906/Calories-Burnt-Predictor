import os
import pandas as pd
from src.logger import logging

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform preprocessing for numerical dataset:
    - Handle missing values
    - Validate ranges
    - Drop duplicates
    """
    try:
        logging.info("Starting preprocessing...")

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Handle missing values (optional: fill/mean/median instead)
        df.dropna(inplace=True)

        # Example: Remove outliers or invalid values
        df = df[df["Heart_Rate"].between(30, 220)]  # realistic range
        df = df[df["Body_Temp"].between(35, 42)]    # °C range
        df = df[df["Duration"] > 0]                 # remove 0-sec workouts

        logging.info(f"Finished preprocessing — final shape: {df.shape}")
        return df

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def main():
    try:
        # Load data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info("Train/test data loaded successfully")

        # Preprocess
        train_processed = preprocess_dataframe(train_data)
        test_processed = preprocess_dataframe(test_data)

        # Save to interim folder
        interim_dir = os.path.join("data", "interim")
        os.makedirs(interim_dir, exist_ok=True)
        train_processed.to_csv(os.path.join(interim_dir, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(interim_dir, "test_processed.csv"), index=False)

        logging.info(f"Processed data saved to {interim_dir}")

    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
