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
    """
    A Conv2d layer wrapper that duplicates selected output channels
    to increase redundancy (hardening).
    """
    def __init__(self, conv, dup_idx):
        """
        Initializes a hardened Conv2d layer by duplicating
        the specified output channel indices.
        """
        super().__init__()
        self.dup_idx = dup_idx
        self.original_out = conv.out_channels

        weight = conv.weight.data
        dup_w = weight[dup_idx]

        new_weight = torch.cat([weight, dup_w], dim=0)

        self.conv = nn.Conv2d(
            conv.in_channels,
            new_weight.size(0),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None)
        )

        self.conv.weight = nn.Parameter(new_weight)
        if conv.bias is not None:
            self.conv.bias = conv.bias

    def forward(self, x):
        """
        Performs forward propagation using the modified convolution layer.
        """
        return self.conv(x)


class HardenedLinear(nn.Module):
    """
    A Linear layer wrapper that duplicates selected output neurons
    to improve robustness.
    """
    def __init__(self, fc, dup_idx):
        """
        Initializes a hardened Linear layer by duplicating
        the specified output neuron indices.
        """
        super().__init__()
        weight = fc.weight.data
        dup_w = weight[dup_idx]

        new_weight = torch.cat([weight, dup_w], dim=0)

        self.fc = nn.Linear(fc.in_features,
                            new_weight.size(0),
                            bias=(fc.bias is not None))

        self.fc.weight = nn.Parameter(new_weight)
        if fc.bias is not None:
            self.fc.bias = fc.bias

    def forward(self, x):
        """
        Performs forward propagation using the modified linear layer.
        """
        return self.fc(x)


def apply_duplication(model, selected):
    """
    Applies channel/neuron duplication to selected layers of the model
    by replacing them with hardened versions.
    """

    model = copy.deepcopy(model)

    for name, module in model.named_modules():

        if name in selected:

            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)

            if isinstance(module, nn.Conv2d):
                setattr(parent, parts[-1],
                        HardenedConv2d(module, selected[name]))

            elif isinstance(module, nn.Linear):
                setattr(parent, parts[-1],
                        HardenedLinear(module, selected[name]))

    return model
