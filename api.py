import base64
import io
from typing import Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification


app = FastAPI(title="Food Classification API")

MODEL_NAME = "nateraw/food"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()


class PredictRequest(BaseModel):
    image: str  # base64 encoded image


def decode_base64_image(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def image_quality_check(image: Image.Image) -> Dict[str, Any]:
    arr = np.array(image)
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
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, 3)

    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        label = model.config.id2label[idx.item()]
        predictions.append({
            "label": label,
            "confidence": round(float(prob.item()), 4)
        })

    return {
        "label": predictions[0]["label"],
        "confidence": predictions[0]["confidence"],
        "top_3": predictions
    }


def route_decision(quality: Dict[str, Any], confidence: float) -> str:
    if not quality["passed"]:
        return "human_review_image_quality"
    if confidence >= 0.85:
        return "auto_approve"
    if confidence >= 0.60:
        return "approve_but_audit"
    return "human_review_low_confidence"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
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