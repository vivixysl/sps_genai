"""Part 2: train AssignmentCNN on CIFAR-10 and save the weights for the API.

CIFAR-10 images are 32x32, but the required architecture expects 64x64 input,
so every image is resized to 64x64 before going into the model.
Run from the project root:  python train_cifar10.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.cnn_model import AssignmentCNN

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = AssignmentCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()            # takes raw logits
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Epoch {epoch + 1}/{EPOCHS}  "
          f"train loss {running_loss / len(train_set):.4f}  test acc {correct / total:.2%}")

torch.save(model.state_dict(), "app/cnn_cifar10.pth")
print("Saved weights to app/cnn_cifar10.pth")
