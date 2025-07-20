import functions_framework
import os
import time
from google.cloud import run_v2, aiplatform, storage

# Read variables set in the Cloud Function's environment
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
PROJECT_NUMBER = os.environ.get("GCP_PROJECT_NUMBER")
REGION = "us-central1"
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")

@functions_framework.cloud_event
def orchestrate_ml_workflow(cloud_event):
    if not all([PROJECT_ID, PROJECT_NUMBER, BUCKET_NAME]):
        raise ValueError("Environment variables must be set.")

    data = cloud_event.data
    file_name = data.get("name")
    
    if not file_name or not file_name.startswith("data/"):
        print(f"File {file_name} is not in the data folder. Skipping.")
        return

    data_path = f"gs://{BUCKET_NAME}/{file_name}"
    timestamp = int(time.time())
    job_name = f"delivery-training-job-{timestamp}"
    model_output_dir = f"gs://{BUCKET_NAME}/models/{job_name}"

    print("--- Starting MLOps Workflow ---")
    
    run_client = run_v2.JobsClient()
    job_parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    job_resource_name = f"{job_parent}/jobs/{job_name}"

    # --- Step 1: Trigger Cloud Run Training Job ---
    try:
        container_args = ["--data-path", data_path, "--model-dir", model_output_dir]
        service_account_email = f"{PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

        job_config = { "template": { "template": { "containers": [{"image": f"gcr.io/{PROJECT_ID}/food-delivery-trainer:latest", "args": container_args}], "service_account": service_account_email } } }

        print(f"Creating and running Cloud Run job: {job_name}")
        run_client.create_job(parent=job_parent, job=job_config, job_id=job_name).result()
        run_operation = run_client.run_job(name=job_resource_name)
        run_operation.result()
        print(f"Training job {job_name} completed successfully.")
    except Exception as e:
        print(f"Error during training job: {e}")
        raise
    finally:
        try:
            print(f"Cleaning up and deleting job: {job_name}")
            run_client.delete_job(name=job_resource_name).result()
            print(f"Successfully deleted job: {job_name}")
        except Exception as e:
            print(f"Cleanup failed for job {job_name}: {e}.")

    # --- Step 2: Deploy Model to Vertex AI ---
    try:
        print(f"Starting deployment of model from {model_output_dir}")
        aiplatform.init(project=PROJECT_ID, location=REGION)
        endpoint_name = "food-delivery-endpoint-final"

        # Use the pre-built scikit-learn container for serving
        serving_container_image = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest"

        registered_model = aiplatform.Model.upload(
            display_name=f"food-delivery-predictor-{timestamp}",
            artifact_uri=model_output_dir,
            serving_container_image_uri=serving_container_image
        )
        
        endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
        if endpoints:
            endpoint = endpoints[0]
            print(f"Existing endpoint found. Undeploying all models.")
            endpoint.undeploy_all(sync=True)
            print("All models undeployed successfully.")
        else:
            print("No existing endpoint found. Creating new endpoint.")
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        
        print(f"Deploying new model {registered_model.resource_name} to endpoint.")
        endpoint.deploy(
            model=registered_model,
            deployed_model_display_name=f"v-{timestamp}",
            machine_type="n1-standard-2",
        )
        
        print(f"Deployment to endpoint {endpoint.resource_name} successful.")
    except Exception as e:
        print(f"Error during deployment: {e}")
        raise

    print("--- MLOps Workflow Finished ---")
