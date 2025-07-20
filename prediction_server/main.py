import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from google.cloud import storage

app = Flask(__name__)

# --- Model Loading ---
# Vertex AI sets an environment variable, AIP_STORAGE_URI,
# which points to the GCS directory containing our model.
model = None
model_dir = os.environ.get("AIP_STORAGE_URI")
if model_dir:
    # The model file is inside this directory
    model_path = os.path.join(model_dir.replace('gs://', '/gcs/'), "model.joblib")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load model from {model_path}. Error: {e}")
else:
    print("FATAL: AIP_STORAGE_URI environment variable not set. Model not loaded.")


# Vertex AI requires a health check route.
@app.route(os.environ.get("AIP_HEALTH_ROUTE", "/health"), methods=["GET"])
def health_check():
    """Health check endpoint required by Vertex AI."""
    return "OK", 200


# Vertex AI requires a prediction route.
@app.route(os.environ.get("AIP_PREDICT_ROUTE", "/predict"), methods=["POST"])
def predict():
    """Prediction endpoint required by Vertex AI."""
    if model is None:
        return "Model not loaded", 500

    try:
        # The request body is a JSON object with an "instances" key.
        request_json = request.get_json()
        instances = request_json["instances"]
        
        # Convert the list of instances into a pandas DataFrame.
        df = pd.DataFrame(instances)
        
        # Make predictions.
        predictions = model.predict(df)
        
        # Format the response as required by Vertex AI.
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return f"Error during prediction: {e}", 400


if __name__ == "__main__":
    # This is used for local testing.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))