import io
import json

import torch
from PIL import Image
from torchvision import transforms

from fashion_model import CLASSES, FashionCNN


def model_fn(model_dir):
    model = FashionCNN()
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location="cpu"))
    model.eval()
    return model


def input_fn(request_body, content_type):
    if content_type in ("image/png", "image/jpeg"):
        image = Image.open(io.BytesIO(request_body)).convert("L")

        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.Lambda(lambda x: 1.0 - x),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        tensor = transform(image).unsqueeze(0)
        return tensor
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_tensor, model):
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    return {"class": CLASSES[pred], "confidence": float(probs[pred])}


def output_fn(prediction, accept):
    accept = accept or "application/json"
    if accept == "application/json":
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
