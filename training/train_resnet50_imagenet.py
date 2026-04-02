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
from tqdm import tqdm

from model import load_resnet50

# ─────────────────────────────────────────────
# 1. SETUP & DOWNLOAD
# ─────────────────────────────────────────────
def prepare_tiny_imagenet(dest="./data"):
    dataset_path = os.path.join(dest, "tiny-imagenet-200")
    if os.path.exists(dataset_path): return dataset_path

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

    # Reorganize Val folder
    val_dir = os.path.join(dataset_path, 'val')
    img_dir = os.path.join(val_dir, 'images')
    if os.path.exists(img_dir):
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

# ─────────────────────────────────────────────
# 2. CONFIG (OPTIMIZED FOR 2x T4 GPUs)
# ─────────────────────────────────────────────
DATA_DIR     = prepare_tiny_imagenet()
NUM_CLASSES  = 200
# ⚡ DOUBLED BATCH SIZE (64 -> 128) for 2 GPUs
BATCH_SIZE   = 128
NUM_EPOCHS   = 10
BASE_LR      = 0.1
SAVE_BEST    = "resnet50_tiny_best.pth"
device       = "cuda" if torch.cuda.is_available() else "cpu"

NORM_MEAN = [0.480, 0.448, 0.397]
NORM_STD  = [0.230, 0.226, 0.226]

# ─────────────────────────────────────────────
# 3. DATALOADERS
# ─────────────────────────────────────────────
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    # T.RandomRotation(15), # Disabled to reduce CPU load slightly
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

train_ds = dsets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_ds   = dsets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_transform)

# ⚡ WORKERS=2 (Fixes the warning you saw)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ─────────────────────────────────────────────
# 4. MODEL SETUP (MULTI-GPU ENABLED)
# ─────────────────────────────────────────────
print("⚙️ Setting up model...")
model = load_resnet50(num_classes=NUM_CLASSES, pth_path=None, is_tiny=True)

# ⚡ MAGIC LINE FOR 2 GPUS
if torch.cuda.device_count() > 1:
    print(f"🔥 Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 9], gamma=0.1)
scaler = torch.amp.GradScaler('cuda')

# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────
def get_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

train_acc_history, val_acc_history = [], []
best_acc = 0.0

print(f"🚀 Starting Training on {device}...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Track accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        loop.set_postfix(loss=loss.item(), acc=100 * correct_train / total_train)
    
    scheduler.step()
    
    # Validation
    epoch_train_acc = 100 * correct_train / total_train
    print("⏳ Validating...")
    val_acc = get_accuracy(model, val_loader)
    
    train_acc_history.append(epoch_train_acc)
    val_acc_history.append(val_acc)
    
    print(f"✅ Epoch [{epoch+1}] | Train: {epoch_train_acc:.2f}% | Val: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), SAVE_BEST)
        print(f"💾 Saved Best Model ({best_acc:.2f}%)")

print("\n🎉 Done Training!")

# ─────────────────────────────────────────────
# 6. PLOTTING RESULTS
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.title("Training vs Validation Accuracy (tiny imagenet on ResNet-50) ")
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_comparison.png")  # Save to file
plt.show()                              # Show in notebook

