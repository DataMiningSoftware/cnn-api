import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from flowers import FLOWER_CLASSES


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 on an image dataset.")
    parser.add_argument("--data", default=None, help="Image folder (class subfolders). If omitted, downloads Flowers-102.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze", action="store_true",
                        help="Fine-tune the feature extractor too (slower, higher accuracy).")
    parser.add_argument("--out", default="model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if args.data:
        dataset = datasets.ImageFolder(args.data, transform=transform)
        classes = dataset.classes
    else:
        Path("./data").mkdir(exist_ok=True)
        dataset = datasets.Flowers102(root="./data", split="train", download=True, transform=transform)
        classes = FLOWER_CLASSES

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Classes: {len(classes)}")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if args.unfreeze:
        for p in model.features.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(
            [
                {"params": model.features.parameters(), "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr},
            ]
        )
    else:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            loss_sum += loss.item() * labels.size(0)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={loss_sum / total:.4f}  acc={correct / total:.4f}")

    torch.save({"state_dict": model.state_dict(), "classes": classes}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
