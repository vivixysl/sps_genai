import io

import torch
import torchvision.transforms as transforms
from PIL import Image

from app.cnn_model import SimpleCNN

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Same preprocessing as in training: resize to 64x64, convert to tensor
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])


class ImageClassifier:
    def __init__(self, weights_path="app/cnn_cifar10.pth"):
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()  # evaluation mode

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = transform(image).unsqueeze(0)  # add batch dimension: 1 x 3 x 64 x 64
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        return {
            "predicted_class": CLASSES[predicted.item()],
            "confidence": round(confidence.item(), 4),
        }
