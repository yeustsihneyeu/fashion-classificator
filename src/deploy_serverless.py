import os

import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig

session = sagemaker.Session()
region = session.boto_region_name

account_id = (
    os.environ.get("ACCOUNT_ID") or boto3.client("sts").get_caller_identity()["Account"]
)

role = os.environ["SAGEMAKER_PIPELINE_ROLE_ARN"]

model_data = f"s3://yeustsihneyeu-fashion/pipelines-h3htitdtpd20-TrainModel-Sx40s9zGcu/output/model.tar.gz"

inference_image = (
    f"{account_id}.dkr.ecr.{region}.amazonaws.com/fashion-inference:latest"
)

inference_model = Model(
    image_uri=inference_image,
    model_data=model_data,
    role=role,
    sagemaker_session=session,
)

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,
    max_concurrency=2,
)

predictor = inference_model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name="fashion-serverless-endpoint",
)

print("✅ Deployed endpoint:", predictor.endpoint_name)
