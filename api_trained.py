import base64
import io
from typing import Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from torchvision import models, transforms


MODEL_PATH = "saved_models/resnet18_food101_acc_0.6585.pth"
NUM_CLASSES = 101

app = FastAPI(title="Food-101 Classification API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictRequest(BaseModel):
    image: str  # base64 encoded image


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    idx_to_class = checkpoint["idx_to_class"]

    # keys may load as strings depending on save/load behavior
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    return model, idx_to_class


model, idx_to_class = load_model()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def decode_base64_image(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def image_quality_check(image: Image.Image) -> Dict[str, Any]:
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    width, height = image.size

    issues = []

    if blur_score < 80:
        issues.append("Image may be blurry")
    if brightness < 40:
        issues.append("Image may be too dark")
    if brightness > 220:
        issues.append("Image may be overexposed")
    if width < 224 or height < 224:
        issues.append("Image resolution may be too low")

    return {
        "passed": len(issues) == 0,
        "blur_score": round(float(blur_score), 2),
        "brightness": round(brightness, 2),
        "width": width,
        "height": height,
        "issues": issues,
    }


def classify_food(image: Image.Image) -> Dict[str, Any]:
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, 3)

    top_3 = []
    for prob, idx in zip(top_probs, top_indices):
        idx_int = int(idx.item())
        top_3.append({
            "label": idx_to_class[idx_int],
            "confidence": round(float(prob.item()), 4)
        })

    return {
        "label": top_3[0]["label"],
        "confidence": top_3[0]["confidence"],
        "top_3": top_3
    }


def route_decision(quality: Dict[str, Any], confidence: float) -> str:
    if not quality["passed"]:
        return "human_review_image_quality"
    if confidence >= 0.70:
        return "auto_approve"
    if confidence >= 0.40:
        return "approve_but_audit"
    return "human_review_low_confidence"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Required workshop endpoint.
    Must return only label + confidence.
    """
    image = decode_base64_image(request.image)
    prediction = classify_food(image)

    return {
        "label": prediction["label"],
        "confidence": prediction["confidence"]
    }


@app.post("/predict_full")
def predict_full(request: PredictRequest):
    """
    Full consulting demo endpoint with human-in-the-loop routing.
    """
    image = decode_base64_image(request.image)

    quality = image_quality_check(image)
    prediction = classify_food(image)
    route = route_decision(quality, prediction["confidence"])

    return {
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "top_3": prediction["top_3"],
        "image_quality": quality,
        "route": route
    }