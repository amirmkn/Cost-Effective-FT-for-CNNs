import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import requests
import zipfile

from model import load_resnet50

# TINY-IMAGENET DOWNLOAD & PREP
def prepare_tiny_imagenet(dest="./data"):
    dataset_path = os.path.join(dest, "tiny-imagenet-200")
    if os.path.exists(dataset_path):
        return dataset_path

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "tiny-imagenet.zip")
    
    print("📥 Downloading Tiny-ImageNet...")
    r = requests.get(url, stream=True)
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: f.write(chunk)
    
    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest)
    os.remove(zip_path)

    # Reorganize Val folder for ImageFolder compatibility
    val_dir = os.path.join(dataset_path, 'val')
    img_dir = os.path.join(val_dir, 'images')
    if os.path.exists(img_dir): # Only run if not already reorganized
        print("📂 Reorganizing validation set...")
        with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
            for line in f.readlines():
                parts = line.split('\t')
                img, cls = parts[0], parts[1]
                target_dir = os.path.join(val_dir, cls)
                os.makedirs(target_dir, exist_ok=True)
                os.rename(os.path.join(img_dir, img), os.path.join(target_dir, img))
        os.rmdir(img_dir)
    return dataset_path

# CONFIG
DATA_DIR     = prepare_tiny_imagenet()
NUM_CLASSES  = 200
BATCH_SIZE   = 64
NUM_EPOCHS   = 10
BASE_LR      = 0.1
SAVE_BEST    = "resnet50_tiny_best.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

NORM_MEAN = [0.480, 0.448, 0.397]
NORM_STD  = [0.230, 0.226, 0.226]

# Standard Tiny-ImageNet transforms
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# ─────────────────────────────────────────────
# DATASET & DATALOADER
# ─────────────────────────────────────────────
train_ds = dsets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_ds   = dsets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ─────────────────────────────────────────────
# MODEL & TOOLS
# ─────────────────────────────────────────────
model = load_resnet50(num_classes=NUM_CLASSES, pth_path=None, is_tiny=True).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 13], gamma=0.1)

# ─────────────────────────────────────────────
# ACCURACY CALCULATOR
# ─────────────────────────────────────────────
def get_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
train_acc_history = []
val_acc_history = []
best_acc = 0.0

print(f"🚀 Training ResNet-50 on Tiny-ImageNet ({device})...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    
    scheduler.step()

    # Calculate Accuracies
    train_acc = get_accuracy(model, train_loader)
    val_acc   = get_accuracy(model, val_loader)
    
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    
    print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Save Best
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_BEST)

# ─────────────────────────────────────────────
# PLOT GENERATION
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_EPOCHS + 1), train_acc_history, label='Training Accuracy', marker='o')
plt.plot(range(1, NUM_EPOCHS + 1), val_acc_history, label='Validation Accuracy', marker='s')
plt.title('ResNet-50 Accuracy: Training vs Validation')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_comparison.png")
plt.show()

print(f"\n✅ Training complete. Best Validation Accuracy: {best_acc:.2f}%")