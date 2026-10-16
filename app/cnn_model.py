"""Part 1: CNN that matches the architecture in Assignment 2."""
import torch
import torch.nn as nn


class AssignmentCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),   # -> 16 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 16 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # -> 32 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 32 x 16 x 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # -> 32*16*16 = 8192
            nn.Linear(32 * 16 * 16, 100),    # -> 100
            nn.ReLU(),
            nn.Linear(100, num_classes),     # -> 10 (logits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


if __name__ == "__main__":
    # Sanity check: a batch of 4 RGB 64x64 images -> 4 x 10 logits
    model = AssignmentCNN()
    out = model(torch.randn(4, 3, 64, 64))
    print(model)
    print("Output shape:", out.shape)  # torch.Size([4, 10])
