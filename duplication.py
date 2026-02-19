import torch
import torch.nn as nn
import copy


def select_top_channels(vuln_dict, hardening_ratio=0.1):
    """
    Selects the top vulnerable channel indices for each layer
    based on the given hardening ratio.
    """
    selected = {}

    for layer, data in vuln_dict.items():
        n = len(data["scores"])
        k = int(n * hardening_ratio)
        selected[layer] = data["sorted_idx"][:k]

    return selected

class HardenedConv2d(nn.Module):
    def __init__(self, conv, bn, dup_idx):
        super().__init__()
        self.dup_idx = [i for i in dup_idx if i < conv.out_channels]
        self.conv = copy.deepcopy(conv)
        self.dup_conv = copy.deepcopy(conv)
        self.bn = bn

    def forward(self, x):
        out_main = self.conv(x)
        out_dup = self.dup_conv(x)
        if len(self.dup_idx) > 0:
            out_main[:, self.dup_idx, :, :] = (
                out_main[:, self.dup_idx, :, :] + out_dup[:, self.dup_idx, :, :]
            ) / 2.0
        if self.bn is not None:
            out_main = self.bn(out_main)
        return out_main


class HardenedLinear(nn.Module):
    def __init__(self, fc, dup_idx):
        super().__init__()
        dup_idx = [i for i in dup_idx if i < fc.out_features]
        if len(dup_idx) == 0:
            self.fc = fc
        else:
            weight = fc.weight.data
            dup_w = weight[dup_idx]
            new_weight = torch.cat([weight, dup_w], dim=0)
            self.fc = nn.Linear(fc.in_features, new_weight.size(0), bias=(fc.bias is not None))
            self.fc.weight = nn.Parameter(new_weight)
            if fc.bias is not None:
                dup_b = fc.bias.data[dup_idx]
                self.fc.bias = nn.Parameter(torch.cat([fc.bias.data, dup_b], dim=0))

    def forward(self, x):
        return self.fc(x)


def apply_duplication(model, selected):
    model = copy.deepcopy(model)
    modules = dict(model.named_modules())

    for name, module in modules.items():
        if name in selected:
            if isinstance(module, nn.Conv2d):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                bn_module = None
                if parts[-1].startswith("conv"):
                    bn_name = "bn" + parts[-1][-1]
                    bn_module = getattr(parent, bn_name, None)
                elif parts[-1] == "0" and isinstance(parent, nn.Sequential):
                    if len(parent) > 1 and isinstance(parent[1], nn.BatchNorm2d):
                        bn_module = parent[1]
                setattr(parent, parts[-1], HardenedConv2d(module, bn_module, selected[name]))
            elif isinstance(module, nn.Linear):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], HardenedLinear(module, selected[name]))
    return model
