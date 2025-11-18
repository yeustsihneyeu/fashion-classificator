import os

import boto3
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def preprocess_data(input_dir, output_dir):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = datasets.FashionMNIST(
        root=input_dir, train=True, download=True, transform=transform
    )
    torch.save(dataset, os.path.join(output_dir, "train_dataset.pt"))
    print("✅ Preprocessed data saved!")


if __name__ == "__main__":
    input_dir = os.environ.get("SM_CHANNEL_INPUT", "/opt/ml/processing/input")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/processing/output")
    os.makedirs(output_dir, exist_ok=True)
    preprocess_data(input_dir, output_dir)
