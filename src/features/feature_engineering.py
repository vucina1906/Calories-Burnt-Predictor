import os
import pandas as pd
from src.logger import logging

def load_data(file_path: str) -> pd.DataFrame:
    """Load processed data from CSV."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        raise

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns like User_ID."""
    try:
        if 'User_ID' in df.columns:
            df = df.drop(columns=['User_ID'])
            logging.info("Dropped column: User_ID")
        return df
    except Exception as e:
        logging.error(f"Error dropping User_ID column: {e}")
        raise

def extract_features_and_target(df: pd.DataFrame, target_column: str = "Calories") -> tuple:
    """Split features and target."""
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        logging.info(f"Split features and target. Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    except Exception as e:
        logging.error(f"Error during feature-target split: {e}")
        raise

def save_data(X: pd.DataFrame, y: pd.Series, prefix: str, output_dir: str = "./data/processed") -> None:
    """Save features and target to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        X.to_csv(os.path.join(output_dir, f"{prefix}_features.csv"), index=False)
        y.to_csv(os.path.join(output_dir, f"{prefix}_target.csv"), index=False)
        logging.info(f"Saved {prefix}_features.csv and {prefix}_target.csv to {output_dir}")
    except Exception as e:
        logging.error(f"Error saving feature/target files: {e}")
        raise

def main():
    try:
        train_df = clean_dataframe(load_data("./data/interim/train_processed.csv"))
        test_df = clean_dataframe(load_data("./data/interim/test_processed.csv"))

        X_train, y_train = extract_features_and_target(train_df, target_column="Calories")
        X_test, y_test = extract_features_and_target(test_df, target_column="Calories")

        save_data(X_train, y_train, prefix="train")
        save_data(X_test, y_test, prefix="test")

        logging.info("✅ Feature engineering completed successfully.")

    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        print(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    main()
