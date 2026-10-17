import torch
import torch.nn as nn
import torch.nn.functional as F


# Same structure as SimpleCNN from Module 4 Practical 3,
# changed to match the Assignment 2 specification:
#   input 64x64x3, fully connected layer with 100 units, 10 output classes
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        )  # Input channels = 3, Output channels = 16
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Pooling layer, will half the dimensions
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        )  # Input channels = 16, Output channels = 32
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # Fully connected layer
        self.fc2 = nn.Linear(100, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 3x64x64  -> 16x32x32
        x = self.pool(F.relu(self.conv2(x)))  # 16x32x32 -> 32x16x16
        x = x.view(-1, 32 * 16 * 16)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    net = SimpleCNN()
    print(net)
    test_input = torch.randn(4, 3, 64, 64)  # a batch of 4 RGB images, 64x64
    print("Output shape:", net(test_input).shape)
