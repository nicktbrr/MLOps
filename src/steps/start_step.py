from metaflow import step
import pandas as pd
from google.cloud import storage
import io


def start_step(self):
    """Starting point - load data"""
    print("Loading data...")
    storage_client = storage.Client.create_anonymous_client()
    
    # Get bucket and blob
    bucket_name = "lab7_gcp_tester1"
    blob_name = "boston.csv"

    file_path = 'gs://mlops-lab7-bucket/flavors_of_cacao.csv'

    self.data = pd.read_csv(file_path)
    
    # bucket = storage_client.bucket(bucket_name)
    # blob = bucket.blob(blob_name)
    
    # # Download the data
    # data_bytes = blob.download_as_bytes()
    
    # # Load into pandas
    # df = pd.read_csv(io.BytesIO(data_bytes))
    # self.data = df