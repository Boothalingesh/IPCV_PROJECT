import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# LOAD GENRES (SAFE)
# -----------------------------
if not os.path.exists("genres.json"):
    raise FileNotFoundError(
        "❌ genres.json not found. Run train.py first."
    )

with open("genres.json") as f:
    genres = json.load(f)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(genres))

model.load_state_dict(
    torch.load("movie_genre_model.pth", map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict(image_path, top_k=5):

    if not os.path.exists(image_path):
        print("❌ Image not found:", image_path)
        return

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(img))[0]

    print("\n🎬 Top Predictions:\n")

    topk = torch.topk(probs, top_k)

    for idx, val in zip(topk.indices, topk.values):
        print(f"{genres[idx]} : {float(val):.3f}")

# -----------------------------
# TEST
# -----------------------------
predict("l.webp")