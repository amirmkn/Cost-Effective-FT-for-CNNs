import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        transform=test_transform
    )

    # sample
    subset_size = 500 

    train_indices = torch.randperm(len(trainset))[:subset_size]
    test_indices = torch.randperm(len(testset))[:subset_size]

    trainset = Subset(trainset, train_indices)
    testset = Subset(testset, test_indices)

    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    # Model
    model = resnet50(weights=None)

    state_dict = torch.load("resnet50.pth", map_location=device)

    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)

    model.load_state_dict(state_dict, strict=False)

    model.fc = nn.Linear(model.fc.in_features, 10)

    model = model.to(device)

    # loss , optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training
    for epoch in range(10): 
        model.train()
        total_loss = 0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss {total_loss:.3f}")

    # Evaluate
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

    print("Final CIFAR10 Accuracy:", correct/total)

    torch.save(model.state_dict(), "resnet50_cifar10.pth")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()