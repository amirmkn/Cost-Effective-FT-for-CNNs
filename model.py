import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import alexnet

class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # ✅ CIFAR‑10 feature map size: 256 × 4 × 4 = 4096
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_resnet50_from_pth(dataset_name, pth_path_dict):

    dataset_name = dataset_name.lower()

    if dataset_name == "imagenet":
        num_classes = 1000
    elif dataset_name == "cifar10":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    else:
        raise ValueError("Unknown dataset")

    model = models.resnet50(weights=None)

    if dataset_name in pth_path_dict:
        pth_path = pth_path_dict[dataset_name]
        state = torch.load(pth_path, map_location="cpu")
        
        saved_classes = state["fc.weight"].shape[0]
        
        model.fc = nn.Linear(model.fc.in_features, saved_classes)
        model.load_state_dict(state)

        if saved_classes != num_classes:
            print(f"Replacing FC layer: {saved_classes} → {num_classes}")
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError("No weight file provided for this dataset")

    return model

def load_alexnet(num_classes=10, pth_path=None):
    model = AlexNetCIFAR(num_classes)
    if pth_path:
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    return model

def load_vgg11(num_classes=10, pth_path=None):
    model = models.vgg11(weights=None)

    # Adapt classifier for CIFAR
    model.classifier = nn.Sequential(
        nn.Linear(512, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )

    if pth_path is not None:
        state = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(state)

    return model

def load_vgg16(num_classes=100, pth_path=None):
    model = models.vgg16(weights=None)

    model.classifier = nn.Sequential(
        nn.Linear(512, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )

    if pth_path is not None:
        state = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(state)

    return model


