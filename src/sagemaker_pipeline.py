import os

import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.processing import Processor
from sagemaker.pytorch import PyTorch
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

session = PipelineSession()

region = os.environ.get("AWS_REGION", session.boto_region_name)
account_id = (
    os.environ.get("ACCOUNT_ID") or boto3.client("sts").get_caller_identity()["Account"]
)
role = os.environ["SAGEMAKER_PIPELINE_ROLE_ARN"]

preprocess_image = (
    f"{account_id}.dkr.ecr.{region}.amazonaws.com/fashion-classificator:latest"
)
inference_image = (
    f"{account_id}.dkr.ecr.{region}.amazonaws.com/fashion-classificator:latest"
)

# --- Step 1: Preprocessing ---
preprocess_processor = Processor(
    image_uri=preprocess_image,
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    sagemaker_session=session,
)

preprocess_step = ProcessingStep(
    name="PreprocessFashionData",
    processor=preprocess_processor,
    outputs=[
        {
            "OutputName": "train",
            "S3Output": "s3://yeustsihneyeu-fashion/fashion/preprocessed/",
        }
    ],
)

# --- Step 2: Training ---
train_estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    role=role,
    instance_type="ml.t3.medium",
    framework_version="2.2",
    py_version="py310",
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=train_estimator,
    inputs={
        "training": preprocess_step.properties.ProcessingOutputConfig.Outputs[
            "train"
        ].S3Output.S3Uri
    },
)

# --- Step 4: Deploy (Serverless Inference) ---
inference_model = Model(
    image_uri=inference_image,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=session,
)

serverless_config = ServerlessInferenceConfig(memory_size_in_mb=2048, max_concurrency=2)

deploy_step_args = inference_model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name="fashion-serverless-endpoint",
)

deploy_step = ModelStep(
    name="DeployServerlessEndpoint",
    step_args=deploy_step_args,
)

# --- Build pipeline ---
pipeline = Pipeline(
    name="FashionMNISTServerlessPipeline",
    steps=[preprocess_step, train_step, deploy_step],
    sagemaker_session=session,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    pipeline.start()
