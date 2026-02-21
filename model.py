import torch
import torch.nn as nn
import torchvision.models as models

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
