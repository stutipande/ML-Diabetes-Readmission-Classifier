# s3_loader.py

import boto3
import joblib
import os

# Your S3 bucket name
BUCKET_NAME = 'diabetes-readmission-models-sp' 

# Files to download from S3
MODEL_FILES = [
    'gradient_boosting_model.pkl',
    'specialty_encoder.pkl',
    'health_index_discretizer.pkl',
    'selected_meds.pkl',
    'expected_columns.pkl'
]

def download_models_from_s3():
    """
    Downloads model files from S3 to local directory if they don't exist
    """
    s3 = boto3.client('s3')
    
    for filename in MODEL_FILES:
        # Check if file already exists locally
        if not os.path.exists(filename):
            print(f"Downloading {filename} from S3...")
            try:
                s3.download_file(BUCKET_NAME, filename, filename)
                print(f"✓ {filename} downloaded successfully")
            except Exception as e:
                print(f"✗ Error downloading {filename}: {e}")
                raise
        else:
            print(f"✓ {filename} already exists locally")

def load_model(filename):
    """
    Loads a model file (downloads from S3 if needed)
    """
    if not os.path.exists(filename):
        download_models_from_s3()
    
    return joblib.load(filename)