import os

import boto3
import sagemaker
from sagemaker import Session
from sagemaker.model import Model
from sagemaker.processing import Processor
from sagemaker.pytorch import PyTorch
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ModelStep, ProcessingStep, TrainingStep

session = Session()

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
    py_version="py311",
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

# --- Step 3: Register Model ---
model_step = ModelStep(
    name="RegisterModel",
    model=Model(
        image_uri=train_estimator.training_image_uri(),
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
    ),
)

# --- Step 4: Deploy (Serverless Inference) ---
inference_model = Model(
    image_uri=inference_image,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
)

serverless_config = ServerlessInferenceConfig(memory_size_in_mb=2048, max_concurrency=2)

deploy_step = ModelStep(
    name="DeployServerlessEndpoint",
    model=inference_model,
    model_deploy_config=serverless_config,
)

# --- Build pipeline ---
pipeline = Pipeline(
    name="FashionMNISTServerlessPipeline",
    steps=[preprocess_step, train_step, model_step, deploy_step],
    sagemaker_session=session,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    pipeline.start()
