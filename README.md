# CNN Fine-tune + Inference API

Fine-tune a pretrained MobileNetV2 on a custom image dataset, then serve it behind a FastAPI endpoint.

## What it demonstrates

- **Transfer learning** — replaces the classifier head of a pretrained MobileNetV2 and fine-tunes it.
- **Custom datasets** — works with any image-folder dataset (`data/<class>/<image>.jpg`), with Flowers-102 as a zero-config fallback.
- **Model serving** — a FastAPI `/api/predict` endpoint that returns top-3 class predictions with confidence.
- This closes the "classical ML only, no deep learning" gap.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> PyTorch is a large install; use the CPU build unless you have a GPU.

## Train

Custom dataset (recommended for a portfolio story):

```powershell
# Organize images as data/<class_name>/<image>.jpg
.\.venv\Scripts\python.exe train.py --data ./data --epochs 5
```

Or let it download Flowers-102 automatically:

```powershell
.\.venv\Scripts\python.exe train.py --epochs 5
```

This writes `model.pth` (weights + class names).

## Serve

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --port 8002
```

Test it:

```powershell
curl.exe -X POST http://localhost:8002/api/predict -F "file=@path\to\image.jpg"
```

## API

- `GET /api/health` — whether a trained model is loaded
- `POST /api/predict` — multipart image → `{ "predictions": [{ "class": "...", "confidence": 0.99 }] }`

## Caveats

- Training on CPU is slow; a GPU is strongly recommended for real fine-tuning.
- The classifier head is fine-tuned (all layers stay frozen) — a common, fast default. Add `model.features.requires_grad_(True)` to fine-tune deeper layers.
- Accuracy depends entirely on your dataset; Flowers-102 (102 classes) is just a demo baseline.
