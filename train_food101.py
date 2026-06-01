import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score

DATASET_ROOT = "food101/food-101"
TRAIN_TXT = os.path.join(DATASET_ROOT, "meta", "train.txt")
TEST_TXT = os.path.join(DATASET_ROOT, "meta", "test.txt")
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")

NUM_CLASSES = 101
TRAIN_LIMIT = 10000
VAL_LIMIT = 2000
BATCH_SIZE = 16
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("saved_models", exist_ok=True)


class Food101Dataset(Dataset):
    def __init__(self, txt_file, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(txt_file, "r") as f:
            self.samples = [line.strip() for line in f.readlines()]

        classes = sorted(os.listdir(images_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path = self.samples[idx]
        class_name = rel_path.split("/")[0]
        image_path = os.path.join(self.images_dir, rel_path + ".jpg")

        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = Food101Dataset(TRAIN_TXT, IMAGES_DIR, transform)
val_dataset = Food101Dataset(TEST_TXT, IMAGES_DIR, transform)

train_subset = Subset(train_dataset, range(TRAIN_LIMIT))
val_subset = Subset(val_dataset, range(VAL_LIMIT))

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

CHECKPOINT = "saved_models/latest_checkpoint.pth"

if os.path.exists(CHECKPOINT):
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded previous checkpoint.")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Using device: {device}")
print(f"Training on {TRAIN_LIMIT} images")
print(f"Validating on {VAL_LIMIT} images")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy:.4f}")

save_path = f"saved_models/resnet18_food101_acc_{accuracy:.4f}.pth"

checkpoint_data = {
    "model_state_dict": model.state_dict(),
    "class_to_idx": train_dataset.class_to_idx,
    "idx_to_class": train_dataset.idx_to_class,
    "accuracy": accuracy,
}

torch.save(checkpoint_data, CHECKPOINT)
torch.save(checkpoint_data, save_path)

print(f"Saved latest checkpoint to {CHECKPOINT}")
print(f"Saved accuracy checkpoint to {save_path}")