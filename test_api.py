import base64
import requests

image_path = "images.jpg"

with open(image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "http://127.0.0.1:8000/predict_full",
    json={"image": image_base64}
)

print(response.json())