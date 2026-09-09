import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, models, transforms

from flowers import FLOWER_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(train: bool):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total


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

    train_transform = build_transform(train=True)
    eval_transform = build_transform(train=False)

    test_loader = None
    if args.data:
        dataset = datasets.ImageFolder(args.data, transform=train_transform)
        classes = dataset.classes
    else:
        Path("./data").mkdir(exist_ok=True)
        train = datasets.Flowers102(root="./data", split="train", download=True, transform=train_transform)
        val = datasets.Flowers102(root="./data", split="val", download=True, transform=train_transform)
        test = datasets.Flowers102(root="./data", split="test", download=True, transform=eval_transform)
        dataset = ConcatDataset([train, val])
        classes = FLOWER_CLASSES
        test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
        print(f"Training on {len(dataset)} images (train+val), evaluating on {len(test)} test images.")

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

    if test_loader is not None:
        print(f"Test accuracy (held-out): {evaluate(model, test_loader, device):.4f}")

    torch.save({"state_dict": model.state_dict(), "classes": classes}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
