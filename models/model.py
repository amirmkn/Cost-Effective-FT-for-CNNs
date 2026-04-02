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


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

def load_resnet50(num_classes=None, pth_path=None, is_tiny=False):
    """
    Loads ResNet50 with flexible class counts and weight sources.
    
    Args:
        num_classes: Target number of output classes. 
                     Defaults to 200 (Tiny) or 1000 (Standard) if None.
        pth_path: Local path to .pth file. If None and weights are needed, 
                  downloads default ImageNet weights.
        is_tiny: If True, modifies architecture for 64x64 input.
    """
    
    # 1. Handle default class counts
    if num_classes is None:
        num_classes = 200 if is_tiny else 1000

    # 2. Initialize Model
    # If no local path is provided, we can download official pre-trained weights
    if pth_path is None:
        print("No local path provided. Downloading/Loading official ImageNet weights...")
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    else:
        print(f"Loading weights from local path: {pth_path}")
        model = models.resnet50(weights=None)
        state = torch.load(pth_path, map_location="cpu")
        
        # Adjust FC layer to match the checkpoint being loaded
        saved_classes = state["fc.weight"].shape[0]
        model.fc = nn.Linear(model.fc.in_features, saved_classes)
        model.load_state_dict(state)

    # 3. Tiny-ImageNet Architecture Tweak
    # Standard ResNet50 downsamples 224->56 in the first two layers.
    # For 64x64, we prevent it from shrinking the image too much.
    if is_tiny:
        print("Modifying ResNet50 layers for Tiny-ImageNet (64x64 resolution)")
        # Change first conv: smaller kernel/stride to preserve pixels
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove initial maxpool to keep spatial resolution higher for deeper layers
        model.maxpool = nn.Identity()

    # 4. Final Class Adjustment
    # If the current model classes (from weights) don't match our target, swap FC
    if model.fc.out_features != num_classes:
        print(f"Adjusting FC layer: {model.fc.out_features} -> {num_classes}")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

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


