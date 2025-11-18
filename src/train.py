import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import FashionCNN


def main():
    data_path = os.path.join(
        os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
        "train_dataset.pt",
    )
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    dataset = torch.load(data_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = FashionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 15

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{epochs}]  " f"Train Loss: {loss/len(loader):.4f}")

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))


if __name__ == "__main__":
    main()
