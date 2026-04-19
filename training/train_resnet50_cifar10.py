import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Dataset
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    # DataLoader
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # Model
    model = resnet50(weights=None)
    state_dict = torch.load("weights/resnet50.pth", map_location=device)

    # Remove classifier weights
    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)

    model.load_state_dict(state_dict, strict=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    # Tracking lists
    train_acc_history = []
    test_acc_history = []
    loss_history = []
    epochs = 10
    epochs_range = range(1, epochs + 1)
    target_accuracy = 93  # Optional line in plot

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in trainloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(out, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        train_acc = 100 * correct / total
        train_acc_history.append(train_acc)
        loss_history.append(total_loss)

        # Test Accuracy per epoch
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = 100 * correct / total
        test_acc_history.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss {total_loss:.3f} | Train Acc {train_acc:.2f}% | Test Acc {test_acc:.2f}%")

    # Final Accuracy
    print("\nFinal CIFAR-10 Test Accuracy:", test_acc_history[-1])

    # SAVE MODEL
    torch.save(model.state_dict(), "weights/resnet50_cifar10.pth")


    # PLOTTING

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_acc_history, label='Train Accuracy', color='blue')
    plt.plot(epochs_range, test_acc_history, label='Test Accuracy', color='red')
    plt.title('ResNet50 CIFAR-10 Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.show()

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, loss_history, label='Training Loss', color='orange')
    plt.title('ResNet50 CIFAR-10 Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
