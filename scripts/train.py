import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from google.cloud import storage

def train_model(data_path, model_dir):
    """Loads data, trains the model, and saves it to GCS."""
    df = pd.read_csv(data_path)

    # --- Preprocessing Logic ---
    df = df.drop(columns=["Order_ID"])
    X = df.drop("Delivery_Time_min", axis=1)
    y = df["Delivery_Time_min"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

    numerical_transformer = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", RobustScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(transformers=[("num", numerical_transformer, numerical_cols), ("cat", categorical_transformer, categorical_cols)])

    # --- Model Definition ---
    final_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            subsample=1.0, n_estimators=150, min_samples_split=10, min_samples_leaf=1,
            max_features='sqrt', max_depth=3, learning_rate=0.05, random_state=42, verbose=2
        ))
    ])

    print("Starting model training...")
    final_model.fit(X, y)
    print("Training complete.")

    # --- Saving the Model ---
    # 1. Save the model to a temporary local file in the container
    local_model_path = '/tmp/model.joblib'
    joblib.dump(final_model, local_model_path)
    print(f"Model saved locally to {local_model_path}")

    # 2. Upload the model to Google Cloud Storage
    # Parse the GCS path to get bucket and blob name
    bucket_name = model_dir.split('/')[2]
    blob_name = '/'.join(model_dir.split('/')[3:]) + '/model.joblib'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_model_path)
    print(f"Model uploaded to gs://{bucket_name}/{blob_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--model-dir', type=str, required=True, help="GCS directory to save the model file.")
    args = parser.parse_args()
    train_model(args.data_path, args.model_dir)