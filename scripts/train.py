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

def train_model(data_path, model_dir):
    """Loads data, trains the model, and saves it."""
    df = pd.read_csv(data_path)

    # --- Preprocessing Logic ---
    df = df.drop(columns=["Order_ID"])
    X = df.drop("Delivery_Time_min", axis=1)
    y = df["Delivery_Time_min"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # --- Model Definition (using best parameters) ---
    final_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            subsample=1.0,
            n_estimators=150,
            min_samples_split=10,
            min_samples_leaf=1,
            max_features='sqrt',
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            verbose=2
        ))
    ])

    # --- Training ---
    print("Starting model training...")
    final_model.fit(X, y)
    print("Training complete.")

    # --- Saving the Model ---
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_filename = os.path.join(model_dir, 'model.joblib')
    joblib.dump(final_model, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help='GCS path to the training data CSV file.')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('AIP_MODEL_DIR'), help='GCS directory to save the trained model.')
    args = parser.parse_args()
    train_model(args.data_path, args.model_dir)
