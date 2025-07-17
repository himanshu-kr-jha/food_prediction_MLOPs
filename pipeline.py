import os
import time
from kfp.v2 import dsl
from kfp.v2.dsl import (Model, Input, Output, component)
from kfp.v2 import compiler

# Environment variables will be passed by Cloud Build
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")

@component(
    packages_to_install=["google-cloud-run", "google-cloud-storage"],
)
def trigger_cloud_run_training_job(
    project: str,
    region: str,
    bucket: str,
    data_path: str,
    model: Output[Model]
):
    """Triggers a Cloud Run Job for training and waits for it to complete."""
    from google.cloud import run_v2
    import google.api_core.exceptions

    client = run_v2.JobsClient()
    job_name = f"delivery-training-job-{int(time.time())}"
    job_parent = f"projects/{project}/locations/{region}"
    job_resource_name = f"{job_parent}/jobs/{job_name}"
    
    # The GCS directory where the model will be saved by the Cloud Run job
    model_output_dir = f"gs://{bucket}/models/{job_name}"
    
    # Arguments for the training script inside the container
    container_args = [
        "--data-path", data_path,
        "--model-dir", model_output_dir 
    ]

    # Configure the Cloud Run Job
    job_config = {
        "template": {
            "template": {
                "containers": [{
                    "image": f"gcr.io/{project}/food-delivery-trainer:latest",
                    "args": container_args,
                }],
                # Use the default compute service account which has storage access
                "service_account": f"{project}-compute@developer.gserviceaccount.com"
            }
        }
    }

    print(f"Creating Cloud Run job: {job_name}")
    client.create_job(parent=job_parent, job=job_config, job_id=job_name).result()
    print("Job created.")

    print(f"Executing Cloud Run job: {job_name}")
    run_operation = client.run_job(name=job_resource_name)
    print(f"Job execution started. Waiting for completion...")
    run_operation.result() # This waits for the job to finish
    print(f"Job {job_name} completed.")
    
    # Set the output artifact URI to the directory where the model was saved
    model.uri = model_output_dir
    print(f"Model artifact located at: {model.uri}")

@component(
    packages_to_install=["google-cloud-aiplatform"],
)
def deploy_model_to_vertex(
    model: Input[Model],
    project: str,
    region: str,
    endpoint_name: str
):
    """Deploys the trained model to a Vertex AI Endpoint."""
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)
    
    registered_model = aiplatform.Model.upload(
        display_name="food-delivery-predictor-hybrid",
        artifact_uri=model.uri,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest"
    )
    
    try:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    except Exception:
        endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
        endpoint = endpoints[0] if endpoints else aiplatform.Endpoint.create(display_name=endpoint_name)

    registered_model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="v1",
        machine_type="n1-standard-2",
    )

@dsl.pipeline(
    name="hybrid-delivery-pipeline-final",
    pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root"
)
def create_pipeline(
    data_path: str = f"gs://{BUCKET_NAME}/data/Food_Delivery_Times.csv",
    endpoint_name: str = "food-delivery-endpoint-hybrid"
):
    train_task = trigger_cloud_run_training_job(
        project=PROJECT_ID,
        region=REGION,
        bucket=BUCKET_NAME,
        data_path=data_path,
    )
    
    deploy_task = deploy_model_to_vertex(
        model=train_task.outputs["model"],
        project=PROJECT_ID,
        region=REGION,
        endpoint_name=endpoint_name,
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=create_pipeline,
        package_path="pipeline.json"
    )
