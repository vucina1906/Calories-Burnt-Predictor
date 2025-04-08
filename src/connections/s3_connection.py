import boto3
import pandas as pd
from src.logger import logging
from io import StringIO

class s3_operations:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region_name="us-east-1"):
        """
        Initialize the S3 client with AWS credentials.
        """
        self.bucket_name = bucket_name
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region_name
            )
            logging.info("✅ S3 client initialized successfully.")
        except Exception as e:
            logging.error(f"❌ Failed to initialize S3 client: {e}")
            raise

    def fetch_file_from_s3(self, file_key: str) -> pd.DataFrame:
        """
        Fetch a CSV file from S3 and load it into a DataFrame.
        :param file_key: Path to file inside S3 bucket
        :return: pandas DataFrame
        """
        try:
            logging.info(f"📦 Fetching '{file_key}' from S3 bucket '{self.bucket_name}'...")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info(f"✅ Successfully loaded '{file_key}' with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            logging.error(f"❌ Error fetching '{file_key}' from S3: {e}")
            raise
