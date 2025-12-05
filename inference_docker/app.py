import io

import torch
from fastapi import FastAPI, Request
from PIL import Image
from torchvision import transforms

from fashion_model import CLASSES, FashionCNN

app = FastAPI()

model = FashionCNN()
model.load_state_dict(torch.load("/opt/ml/model/model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


@app.get("/ping")
async def ping():
    # health-check для SageMaker
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(request: Request):
    # SageMaker шлёт сырые байты в body
    body = await request.body()

    image = Image.open(io.BytesIO(body)).convert("L")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    return {"class": CLASSES[pred], "confidence": float(probs[pred])}
