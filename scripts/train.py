# import argparse
# import os
# import pandas as pd
# import joblib
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, RobustScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import GradientBoostingRegressor
# from google.cloud import storage

# def train_model(data_path, model_dir):
#     """Loads data, trains the model, and saves it to GCS."""
#     df = pd.read_csv(data_path)

#     # --- Preprocessing Logic ---
#     df = df.drop(columns=["Order_ID"])
#     X = df.drop("Delivery_Time_min", axis=1)
#     y = df["Delivery_Time_min"]

#     categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
#     numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

#     numerical_transformer = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", RobustScaler())])
#     categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
#     preprocessor = ColumnTransformer(transformers=[("num", numerical_transformer, numerical_cols), ("cat", categorical_transformer, categorical_cols)])

#     # --- Model Definition ---
#     final_model = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', GradientBoostingRegressor(
#             subsample=1.0, n_estimators=150, min_samples_split=10, min_samples_leaf=1,
#             max_features='sqrt', max_depth=3, learning_rate=0.05, random_state=42
#         ))
#     ])

#     print("Starting model training...")
#     final_model.fit(X, y)
#     print("Training complete.")

#     # --- Saving the Model ---
#     local_model_path = '/tmp/model.joblib'
#     joblib.dump(final_model, local_model_path)
#     print(f"Model saved locally to {local_model_path}")

#     bucket_name = model_dir.split('/')[2]
#     blob_name = '/'.join(model_dir.split('/')[3:]) + '/model.joblib'

#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(blob_name)

#     blob.upload_from_filename(local_model_path)
#     print(f"Model uploaded to gs://{bucket_name}/{blob_name}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-path', type=str, required=True)
#     parser.add_argument('--model-dir', type=str, required=True, help="GCS directory to save the model file.")
#     args = parser.parse_args()
#     train_model(args.data_path, args.model_dir)

import argparse
import os
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from google.cloud import storage

def train_model(data_path, model_dir):
    """Loads data, trains the model, evaluates it, and saves it to GCS."""
    
    start_time = time.time()
    
    # --- 1. Data Loading ---
    print("--- 1. Data Loading ---")
    t_load_start = time.time()
    df = pd.read_csv(data_path)
    print(f"Data loaded in {time.time() - t_load_start:.2f} seconds. Shape: {df.shape}")

    # --- 2. Preprocessing & Splitting ---
    print("\n--- 2. Preprocessing & Data Splitting ---")
    t_prep_start = time.time()
    df.dropna(subset=['Delivery_Time_min'], inplace=True) # Drop rows where target is null
    
    X = df.drop(columns=["Order_ID", "Delivery_Time_min"])
    y = df["Delivery_Time_min"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows).")
    
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()

    numerical_transformer = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", RobustScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(transformers=[("num", numerical_transformer, numerical_cols), ("cat", categorical_transformer, categorical_cols)])
    print(f"Preprocessing pipelines created in {time.time() - t_prep_start:.2f} seconds.")

    # --- 3. Model Training ---
    print("\n--- 3. Model Training ---")
    t_train_start = time.time()
    final_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            subsample=1.0, n_estimators=150, min_samples_split=10, min_samples_leaf=1,
            max_features='sqrt', max_depth=3, learning_rate=0.05, random_state=42
        ))
    ])
    final_model.fit(X_train, y_train)
    print(f"Model training completed in {time.time() - t_train_start:.2f} seconds.")

    # --- 4. Model Evaluation ---
    print("\n--- 4. Model Evaluation ---")
    t_eval_start = time.time()
    predictions = final_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Evaluation completed in {time.time() - t_eval_start:.2f} seconds.")
    print("-----------------------------------------")
    print(f"  - R-squared Score: {r2:.4f}")
    print(f"  - Mean Absolute Error: {mae:.2f} minutes")
    print("-----------------------------------------")

    # --- 5. Saving the Model ---
    print("\n--- 5. Saving Model ---")
    t_save_start = time.time()
    local_model_path = '/tmp/model.joblib'
    joblib.dump(final_model, local_model_path)
    
    bucket_name = model_dir.split('/')[2]
    blob_name = '/'.join(model_dir.split('/')[3:]) + '/model.joblib'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_model_path)
    print(f"Model uploaded to gs://{bucket_name}/{blob_name} in {time.time() - t_save_start:.2f} seconds.")
    
    total_duration = time.time() - start_time
    print(f"\n--- Training script finished in {total_duration:.2f} seconds ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--model-dir', type=str, required=True, help="GCS directory to save the model file.")
    args = parser.parse_args()
    train_model(args.data_path, args.model_dir)
