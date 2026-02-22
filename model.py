import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import alexnet


def load_resnet50_from_pth(dataset_name, pth_path_dict):

    dataset_name = dataset_name.lower()

    # تعیین تعداد کلاس
    if dataset_name == "imagenet":
        num_classes = 1000
    elif dataset_name == "cifar10":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    else:
        raise ValueError("Unknown dataset")

    # ساخت مدل پایه
    model = models.resnet50(weights=None)

    # اگر برای این دیتاست فایل جدا داری
    if dataset_name in pth_path_dict:
        pth_path = pth_path_dict[dataset_name]
        state = torch.load(pth_path, map_location="cpu")
        
        saved_classes = state["fc.weight"].shape[0]
        
        model.fc = nn.Linear(model.fc.in_features, saved_classes)
        model.load_state_dict(state)

        # اگر کلاس‌ها متفاوت بود (مثلاً ImageNet weight روی CIFAR)
        if saved_classes != num_classes:
            print(f"Replacing FC layer: {saved_classes} → {num_classes}")
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError("No weight file provided for this dataset")

    return model
def load_alexnet(num_classes=10, pth_path=None):
    """
    AlexNet adapted for CIFAR-10 (paper baseline)
    """
    model = models.alexnet(weights=None)

    # Adapt first conv layer for CIFAR (32x32)
    model.features[0] = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1
    )

    # Remove aggressive downsampling
    model.features[2] = nn.Identity()  # remove maxpool
    model.features[5] = nn.Identity()  # remove maxpool

    # Adjust classifier
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 4 * 4, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )

    if pth_path is not None:
        state = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(state)

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


