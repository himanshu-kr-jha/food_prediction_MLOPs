import functions_framework
import os
import time
from google.cloud import run_v2, storage

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
    model_output_dir = f"gs://{BUCKET_NAME}/models/latest"

    print("--- Starting MLOps Workflow ---")
    
    run_client = run_v2.JobsClient()
    job_name = f"delivery-training-job-{int(time.time())}"
    job_parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    job_resource_name = f"{job_parent}/jobs/{job_name}"

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

    try:
        print("Starting deployment of new Cloud Run service revision.")
        service_client = run_v2.ServicesClient()
        service_name = "prediction-server"
        service_path = f"projects/{PROJECT_ID}/locations/{REGION}/services/{service_name}"
        
        current_service = service_client.get_service(name=service_path)
        current_service.template.template.containers[0].image = f"gcr.io/{PROJECT_ID}/food-delivery-predictor:latest"
        
        update_operation = service_client.update_service(service=current_service)
        update_operation.result()
        
        print(f"Successfully deployed new revision to Cloud Run service: {service_name}")
    except Exception as e:
        print(f"Error during Cloud Run service deployment: {e}")
        raise

    print("--- MLOps Workflow Finished ---")