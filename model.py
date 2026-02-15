import torch
import torch.nn as nn
import torchvision.models as models

def load_resnet50_from_pth(pth_path, num_classes):
    model = models.resnet50(weights=None)
    state = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
