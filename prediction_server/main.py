import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from google.cloud import storage

app = Flask(__name__)

model = None
local_model_path = "/tmp/model.joblib"

try:
    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME environment variable not set.")

    model_blob_path = "models/latest/model.joblib"
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_blob_path)
    
    print(f"Downloading model from gs://{bucket_name}/{model_blob_path}...")
    blob.download_to_filename(local_model_path)
    print("Model downloaded successfully.")
    
    model = joblib.load(local_model_path)
    print("Model loaded into memory.")

except Exception as e:
    print(f"FATAL: Could not load model. Predictions will fail. Error: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    return "OK", 200


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded", 500
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return jsonify({"predicted_delivery_time_min": round(prediction[0], 2)})
    except Exception as e:
        return f"Error during prediction: {e}", 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))