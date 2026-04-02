import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model import load_alexnet
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

best_acc = 0.0
target_accuracy = 73.15  # IEEE paper target

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Preprocessing
transform = T.Compose([
    T.Resize(32),
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std  = [0.2023, 0.1994, 0.2010]
    )
])

# Test data should not have augmentation
transform_test = T.Compose([
    T.Resize(32),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

train_ds = dsets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_ds  = dsets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

# Model loading
model = load_alexnet(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Histories for plotting
train_acc_history = []
test_acc_history = []
loss_history = []

print("Starting Training...")

for epoch in range(90):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    # Calculate accuracy and loss for the epoch    
    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc = evaluate_accuracy(model, test_loader, device)
    avg_loss = running_loss / len(train_loader)
    
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    loss_history.append(avg_loss)
    
    print(f"Epoch [{epoch+1}/90] - Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "alexnet_cifar10_best.pth")
        print(f"✨ New best model saved with {best_acc:.2f}% accuracy")

    if test_acc >= target_accuracy:
        print(f"🎯 Target accuracy {target_accuracy}% reached!")

torch.save(model.state_dict(), "alexnet_cifar10_final.pth")
print("✅ Training Finished. Final model saved.")

# Plotting Accuracy and Loss
epochs_range = range(1, len(train_acc_history) + 1)

# Accuracy Plot
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_acc_history, label='Train Accuracy', color='blue')
plt.plot(epochs_range, test_acc_history, label='Test Accuracy', color='red')
plt.axhline(y=target_accuracy, color='green', linestyle='--', label=f'IEEE Target ({target_accuracy}%)')
plt.title('AlexNet CIFAR-10 Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()

# Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, loss_history, label='Training Loss', color='orange')
plt.title('AlexNet CIFAR-10 Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()
