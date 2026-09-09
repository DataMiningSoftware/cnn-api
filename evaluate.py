import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from flowers import FLOWER_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def topk(output, k):
    return output.topk(k).indices.tolist()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned MobileNetV2 on Flowers-102.")
    parser.add_argument("--data", default="./data")
    parser.add_argument("--checkpoint", default="model.pth")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on at most N test images.")
    parser.add_argument("--out-conf", default="confusion_matrix.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test = datasets.Flowers102(root=args.data, split="test", download=True, transform=build_transform())
    if args.limit:
        test = Subset(test, range(min(args.limit, len(test))))
    loader = DataLoader(test, batch_size=64, shuffle=False)

    classes = FLOWER_CLASSES
    n = len(classes)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, n)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    correct = 0
    total = 0
    top5 = 0
    confusion = [[0] * n for _ in range(n)]
    samples = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out = model(images)
            pred = out.argmax(1).cpu()
            labels = labels.cpu()
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            for i in range(labels.size(0)):
                confusion[labels[i].item()][pred[i].item()] += 1
                if len(samples) < 5:
                    probs = out.softmax(1)
                    samples.append(
                        {
                            "true": classes[labels[i].item()],
                            "predicted": classes[pred[i].item()],
                            "confidence": round(float(probs[i].max()), 4),
                        }
                    )

    acc = correct / total
    per_class = [
        (classes[i], confusion[i][i] / max(1, sum(confusion[i])), sum(confusion[i]))
        for i in range(n)
    ]
    per_class_sorted = sorted(per_class, key=lambda x: x[1])
    best = per_class_sorted[-5:][::-1]
    worst = per_class_sorted[:5]

    print(f"\nTest accuracy: {acc:.4f}  ({correct}/{total})")

    print("\nSample predictions:")
    for s in samples:
        print(f"  true={s['true']:<28} predicted={s['predicted']:<28} conf={s['confidence']}")

    print("\nBest classes by accuracy:")
    for name, a, cnt in best:
        print(f"  {name:<28} {a:.2f}  (n={cnt})")
    print("\nWorst classes by accuracy:")
    for name, a, cnt in worst:
        print(f"  {name:<28} {a:.2f}  (n={cnt})")

    with open(args.out_conf, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_class"] + classes)
        for i in range(n):
            w.writerow([classes[i]] + confusion[i])
    print(f"\nConfusion matrix written to {args.out_conf}")

    with open("results.json", "w") as f:
        import json

        json.dump(
            {
                "accuracy": acc,
                "total": total,
                "top_classes": [{"class": c, "accuracy": round(a, 4), "n": cnt} for c, a, cnt in best],
                "worst_classes": [{"class": c, "accuracy": round(a, 4), "n": cnt} for c, a, cnt in worst],
                "samples": samples,
            },
            f,
            indent=2,
        )
    print("Results written to results.json")


if __name__ == "__main__":
    main()
