import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset transforms
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full CIFAR-10 datasets (no subset sampling)
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=test_transform
    )

    # Use full datasets directly in DataLoader
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    # Model: ResNet50 with pretrained weights (optional)
    # If you have a local "resnet50.pth" file, keep the loading as before.
    # Otherwise, use torchvision's built-in weights for better performance.
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # or weights=None for random init
    # Replace the final fully connected layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0.0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss {total_loss:.3f}")

    # Evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    print("Final CIFAR‑10 Accuracy:", correct / total)

    torch.save(model.state_dict(), "resnet50_cifar10.pth")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()