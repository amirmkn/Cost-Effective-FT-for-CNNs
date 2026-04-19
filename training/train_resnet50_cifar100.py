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
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )

    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )

    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    #  Model 
    model = resnet50(weights=None)

    state_dict = torch.load("weights/resnet50.pth", map_location=device)

    # Remove original classifier weights
    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)

    model.load_state_dict(state_dict, strict=False)

    # Replace classifier for CIFAR100
    model.fc = nn.Linear(model.fc.in_features, 100)

    model = model.to(device)

    #  Loss + Optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    #  History storage 
    train_acc_history = []
    test_acc_history = []
    loss_history = []

    epochs = 10
    epochs_range = range(1, epochs + 1)

    target_accuracy = 0  # optional reference line

    #  Training 
    for epoch in range(epochs):

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in trainloader:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Train accuracy
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total

        train_acc_history.append(train_acc)
        loss_history.append(total_loss)

        #  Test evaluation 
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():

            for x, y in testloader:

                x, y = x.to(device), y.to(device)

                out = model(x)
                pred = out.argmax(1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = 100 * correct / total

        test_acc_history.append(test_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss {total_loss:.3f} | "
            f"Train Acc {train_acc:.2f}% | "
            f"Test Acc {test_acc:.2f}%"
        )

    #  Final Accuracy 
    print("\nFinal CIFAR‑100 Test Accuracy:", test_acc_history[-1])

    #  Save model 
    torch.save(model.state_dict(), "../weights/resnet50_cifar100.pth")

    # =========
    # ====     PLOTS     =======
    # =========

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_acc_history, label='Train Accuracy', color='blue')
    plt.plot(epochs_range, test_acc_history, label='Test Accuracy', color='red')

    plt.title('ResNet50 CIFAR‑100 Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.savefig('cifar100_accuracy_plot.png')
    plt.show()

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, loss_history, label='Training Loss', color='orange')

    plt.title('ResNet50 CIFAR‑100 Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(True)

    plt.savefig('cifar100_loss_plot.png')
    plt.show()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
