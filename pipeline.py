import os
from kfp.v2 import dsl
from kfp.v2.dsl import (Model, Input, component)
from kfp.v2 import compiler
from google.cloud import aiplatform

# Get environment variables
PROJECT_ID = os.environ["GCP_PROJECT_ID"]
REGION = os.environ["GCP_REGION"]
BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

# Define the training component
@component(
    base_image=f"gcr.io/{PROJECT_ID}/food-delivery-trainer:latest",
)
def train_model_component(
    data_path: str,
    model: dsl.Output[dsl.Model],
):
    # The base_image already contains the training script.
    # We just need to pass the arguments.
    # The component will execute the ENTRYPOINT of the Docker image.
    pass

# Define the deployment component
@component(
    packages_to_install=["google-cloud-aiplatform"],
)
def deploy_model_component(
    model: Input[Model],
    project: str,
    region: str,
    endpoint_name: str
) -> dsl.OutputPath(str):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    
    # Upload the model to Vertex AI Model Registry
    registered_model = aiplatform.Model.upload(
        display_name="food-delivery-predictor",
        artifact_uri=model.uri,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest"
    )

    # Create an endpoint if it doesn't exist
    try:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    except Exception:
        endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
        endpoint = endpoints[0]

    # Deploy the model to the endpoint
    registered_model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="v1",
        machine_type="n1-standard-2",
    )
    return endpoint.resource_name

# Define the main pipeline
@dsl.pipeline(
    name="food-delivery-prediction-pipeline",
    pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root"
)
def create_pipeline(
    data_path: str = f"gs://{BUCKET_NAME}/data/Food_Delivery_Times.csv",
    endpoint_name: str = "food-delivery-endpoint"
):
    train_task = train_model_component(data_path=data_path)
    
    deploy_task = deploy_model_component(
        model=train_task.outputs["model"],
        project=PROJECT_ID,
        region=REGION,
        endpoint_name=endpoint_name,
    )

# Compile the pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=create_pipeline,
        package_path="pipeline.json"
    )
