import io
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import models, transforms

app = FastAPI(title="Image Classifier API")

MODEL_PATH = Path(__file__).parent / "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

model = None
classes = []


def load_model() -> bool:
    global model, classes
    if not MODEL_PATH.exists():
        return False
    ckpt = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    return True


loaded = load_model()


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": loaded, "num_classes": len(classes)}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not loaded:
        return {"error": "model.pth not found. Run train.py first."}
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    top = probs.topk(3)
    return {
        "predictions": [
            {"class": classes[i], "confidence": round(probs[i].item(), 4)}
            for i in top.indices.tolist()
        ]
    }
