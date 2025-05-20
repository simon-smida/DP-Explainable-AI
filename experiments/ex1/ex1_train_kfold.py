#!/usr/bin/env python3
"""
train_kfold.py: Perform k-fold cross-validation on the "Proper" CNN model for MNIST and save the best fold.
Usage:
    python train_kfold.py --k-folds 5 --epochs 8 --batch-size 128 --lr 1e-3 \
                          --data-dir ./data --save-path ./models/best_model.pt
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

# --- Model Definition ---
class CNNModel(nn.Module):
    def __init__(self, dropout1=0.25, dropout2=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool  = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

# --- Utility Functions ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def load_mnist(data_dir, train):
    return datasets.MNIST(data_dir, train=train, download=True, transform=get_data_transforms())


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss/len(loader), correct/total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct/total

# --- Main k‑fold logic ---
def main(args):
    dataset = load_mnist(args.data_dir, train=True)
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    best_acc, best_state = 0.0, None

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
        print(f"Fold {fold}/{args.k_folds}")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=args.batch_size, shuffle=False)

        model = CNNModel(dropout1=0.25, dropout2=0.25).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            print(f"  Epoch {epoch}: loss={loss:.4f}, acc={train_acc:.4f}")

        val_acc = evaluate(model, val_loader)
        print(f"  Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, best_state = val_acc, model.state_dict()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(best_state, args.save_path)
    print(f"Best model saved: {args.save_path} (val_acc={best_acc:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Fold CV for Proper CNN on MNIST")
    parser.add_argument("--k-folds",    type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=8)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--data-dir",   type=str,   default="./data")
    parser.add_argument("--save-path",  type=str,   default="./models/proper_kfold_best.pt")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    main(args)