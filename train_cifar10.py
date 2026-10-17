import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from app.cnn_model import SimpleCNN

torch.manual_seed(42)

# Check which GPU is available
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

EPOCHS = 5
BATCH_SIZE = 32

# Prepare the Data
# CIFAR-10 images are 32x32, but the assignment's CNN takes 64x64 input,
# so we resize every image to 64x64 before converting it to a tensor.
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SimpleCNN().to(device)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(EPOCHS):
    running_loss = 0.0
    running_correct, running_total = 0, 0
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)
        running_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS}  "
        f"avg loss: {running_loss / len(train_loader):.4f}  "
        f"avg accuracy: {running_correct / running_total:.3f}"
    )
print("Finished Training")

# Evaluation
test_correct = 0
test_total = 0
model.eval()
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_outputs = model(test_images)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_total += test_labels.size(0)
        test_correct += (test_predicted == test_labels).sum().item()
test_accuracy = 100 * test_correct / test_total
print(f"Accuracy of the network on the 10000 test images: {test_accuracy:.2f}%")

# Save the trained weights so the API can load them
torch.save(model.state_dict(), "app/cnn_cifar10.pth")
print("Saved model weights to app/cnn_cifar10.pth")
