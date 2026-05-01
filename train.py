import os
import ast
import json
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_DIR = "dataset/Images"
CSV_PATH = "dataset/train.csv"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LR = 0.0003

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)
df["Genre"] = df["Genre"].apply(ast.literal_eval)

all_genres = sorted(set(g for row in df["Genre"] for g in row))
label_cols = all_genres

print("Genres:", label_cols)

# ✅ SAVE GENRES (important)
with open("genres.json", "w") as f:
    json.dump(label_cols, f)

# Multi-label encoding
for genre in label_cols:
    df[genre] = df["Genre"].apply(lambda x: 1 if genre in x else 0)

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# DATASET
# -----------------------------
class PosterDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        movie_id = str(self.df.loc[idx, "Id"])
        img_path = os.path.join(IMAGE_DIR, movie_id + ".jpg")

        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        labels = torch.tensor(
            self.df.loc[idx, label_cols].values.astype(np.float32)
        )

        return image, labels

train_loader = DataLoader(PosterDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(PosterDataset(val_df), batch_size=BATCH_SIZE)

# -----------------------------
# MODEL
# -----------------------------
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(label_cols))
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAIN LOOP
# -----------------------------
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "movie_genre_model.pth")

print("✅ Training Complete")