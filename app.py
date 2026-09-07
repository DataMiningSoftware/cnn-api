import io
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import models, transforms

from flowers import FLOWER_CLASSES

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
    if len(classes) == len(FLOWER_CLASSES) and all(c.isdigit() for c in classes):
        classes = [FLOWER_CLASSES[int(c)] for c in classes]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    return True


loaded = load_model()

UI = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Image Classifier</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: linear-gradient(135deg, #10162f 0%, #1a1f3a 100%); color: #e8ecf7;
  }
  .card { width: min(520px, 92vw); background: #1c2340; border: 1px solid #2c3560;
    border-radius: 16px; padding: 32px; box-shadow: 0 24px 60px rgba(0,0,0,.4); }
  h1 { margin: 0 0 6px; font-size: 1.5rem; }
  p.sub { margin: 0 0 24px; color: #9aa5cc; font-size: .9rem; }
  .drop {
    border: 2px dashed #3a4575; border-radius: 12px; padding: 36px 16px; text-align: center;
    cursor: pointer; transition: .15s; background: #151b33;
  }
  .drop:hover, .drop.drag { border-color: #6c7cff; background: #171e3a; }
  .drop img { max-width: 100%; max-height: 240px; border-radius: 8px; display: none; }
  .drop img.show { display: inline-block; }
  .drop .hint { color: #9aa5cc; font-size: .9rem; }
  input[type=file] { display: none; }
  button {
    width: 100%; margin-top: 16px; padding: 12px; border: 0; border-radius: 10px; cursor: pointer;
    background: #5b6cff; color: #fff; font-size: 1rem; font-weight: 600; transition: .15s;
  }
  button:disabled { opacity: .5; cursor: not-allowed; }
  button:not(:disabled):hover { background: #6c7cff; }
  #results { margin-top: 20px; display: none; }
  .pred { display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; background: #151b33; border-radius: 10px; margin-bottom: 8px; }
  .pred .cls { font-weight: 600; }
  .pred .conf { color: #7ce38b; font-variant-numeric: tabular-nums; }
  .bar { height: 6px; background: #2c3560; border-radius: 3px; margin-top: 6px; overflow: hidden; }
  .bar span { display: block; height: 100%; background: linear-gradient(90deg,#5b6cff,#7ce38b); }
  #error { margin-top: 16px; color: #ff7a7a; font-size: .9rem; display: none; }
</style>
</head>
<body>
<div class="card">
  <h1>Image Classifier</h1>
  <p class="sub">Fine-tuned MobileNetV2 &mdash; drop an image to get top-3 predictions.</p>
  <div class="drop" id="drop">
    <img id="preview" alt="preview" />
    <div class="hint" id="hint">Click or drag an image here</div>
  </div>
  <input type="file" id="file" accept="image/*" />
  <button id="btn" disabled>Classify</button>
  <div id="error"></div>
  <div id="results"></div>
</div>
<script>
  const drop = document.getElementById('drop');
  const file = document.getElementById('file');
  const preview = document.getElementById('preview');
  const hint = document.getElementById('hint');
  const btn = document.getElementById('btn');
  const results = document.getElementById('results');
  const error = document.getElementById('error');

  let currentFile = null;

  drop.onclick = () => file.click();
  drop.ondragover = e => { e.preventDefault(); drop.classList.add('drag'); };
  drop.ondragleave = () => drop.classList.remove('drag');
  drop.ondrop = e => {
    e.preventDefault(); drop.classList.remove('drag');
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  };
  file.onchange = () => setFile(file.files[0]);

  function setFile(f) {
    if (!f) return;
    currentFile = f;
    const reader = new FileReader();
    reader.onload = e => { preview.src = e.target.result; preview.classList.add('show'); hint.style.display = 'none'; };
    reader.readAsDataURL(f);
    btn.disabled = false;
  }

  btn.onclick = async () => {
    const f = currentFile;
    if (!f) return;
    btn.disabled = true; btn.textContent = 'Classifying...';
    error.style.display = 'none'; results.style.display = 'none';
    const fd = new FormData();
    fd.append('file', f);
    try {
      const res = await fetch('/api/predict', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Request failed');
      const max = data.predictions[0].confidence;
      results.innerHTML = data.predictions.map(p => `
        <div class="pred">
          <div style="flex:1">
            <div style="display:flex;justify-content:space-between">
              <span class="cls">${p.class}</span>
              <span class="conf">${(p.confidence * 100).toFixed(2)}%</span>
            </div>
            <div class="bar"><span style="width:${(p.confidence / max) * 100}%"></span></div>
          </div>
        </div>`).join('');
      results.style.display = 'block';
    } catch (err) {
      error.textContent = err.message; error.style.display = 'block';
    } finally {
      btn.disabled = false; btn.textContent = 'Classify';
    }
  };
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return UI


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
