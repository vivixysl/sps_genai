"""Loads the trained CNN once and classifies a single image."""
import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.cnn_model import AssignmentCNN

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
WEIGHTS = Path(__file__).parent / "cnn_cifar10.pth"

_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


class ImageClassifier:
    def __init__(self, weights_path: Path = WEIGHTS):
        self.model = AssignmentCNN(num_classes=10)
        if weights_path.exists():
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()  # inference mode

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = _transform(image).unsqueeze(0)          # 1 x 3 x 64 x 64
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(probs.argmax())
        return {"predicted_class": CLASSES[idx], "confidence": round(float(probs[idx]), 4)}
