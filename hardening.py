import torch
import torch.nn as nn
import copy

# =========================================================
# Utility
# =========================================================

def get_parent(model, name):
    components = name.split(".")
    parent = model
    for c in components[:-1]:
        parent = getattr(parent, c)
    return parent

def get_last_linear_name(model):
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
    return last_name

# =========================================================
# 1️⃣ Channel Duplication Wrapper
# =========================================================

class DuplicatedLayer(nn.Module):
    """
    Wraps Conv2d or Linear layer to duplicate specified channels/neurons.
    """
    def __init__(self, layer, vulnerable_idx):
        super().__init__()
        self.layer = layer
        self.vulnerable_idx = vulnerable_idx

        # Duplicate parameters
        if isinstance(layer, nn.Conv2d):
            out_channels = len(vulnerable_idx)
            self.duplicate = nn.Conv2d(
                layer.in_channels,
                out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=(layer.bias is not None)
            )
            self.duplicate.weight.data = layer.weight.data[vulnerable_idx].clone()
            if layer.bias is not None:
                self.duplicate.bias.data = layer.bias.data[vulnerable_idx].clone()
        elif isinstance(layer, nn.Linear):
            out_features = len(vulnerable_idx)
            self.duplicate = nn.Linear(layer.in_features, out_features, bias=(layer.bias is not None))
            self.duplicate.weight.data = layer.weight.data[vulnerable_idx].clone()
            if layer.bias is not None:
                self.duplicate.bias.data = layer.bias.data[vulnerable_idx].clone()
        else:
            raise ValueError("Layer type not supported for duplication")

    def forward(self, x):
        main_out = self.layer(x)
        dup_out = self.duplicate(x)
        return main_out, dup_out

# =========================================================
# 2️⃣ EDAC Layer
# =========================================================

class EDACLayer(nn.Module):
    def __init__(self, min_vals, max_vals, vulnerable_idx):
        super().__init__()
        self.register_buffer("min_vals", min_vals)
        self.register_buffer("max_vals", max_vals)
        self.vulnerable_idx = vulnerable_idx

    def forward(self, main_out, dup_out):
        main_out = torch.nan_to_num(main_out, nan=0.0)
        dup_out = torch.nan_to_num(dup_out, nan=0.0)
        out = main_out.clone()

        # Duplicated channels
        for i, ch in enumerate(self.vulnerable_idx):
            m = main_out[:, ch]
            d = dup_out[:, i]
            min_v = self.min_vals[ch]
            max_v = self.max_vals[ch]

            m_valid = (m >= min_v) & (m <= max_v)
            d_valid = (d >= min_v) & (d <= max_v)

            both_valid = m_valid & d_valid & (m != d)
            out[:, ch][both_valid] = torch.minimum(m[both_valid], d[both_valid])

            only_m = m_valid & (~d_valid)
            only_d = d_valid & (~m_valid)
            out[:, ch][only_m] = m[only_m]
            out[:, ch][only_d] = d[only_d]

            none_valid = (~m_valid) & (~d_valid)
            out[:, ch][none_valid] = 0

        # Non-duplicated channels
        for ch in range(out.shape[1]):
            if ch not in self.vulnerable_idx:
                invalid = (out[:, ch] < self.min_vals[ch]) | (out[:, ch] > self.max_vals[ch])
                out[:, ch][invalid] = 0

        return out

# =========================================================
# 3️⃣ Hardened Block (DuplicatedLayer + EDACLayer)
# =========================================================

class HardenedBlock(nn.Module):
    def __init__(self, dup_layer, edac_layer):
        super().__init__()
        self.dup_layer = dup_layer
        self.edac = edac_layer

    def forward(self, x):
        main_out, dup_out = self.dup_layer(x)
        out = self.edac(main_out, dup_out)
        return out

# =========================================================
# 4️⃣ Profiling Detection Intervals
# =========================================================

def profile_model(model, loader, device):
    model.eval()
    min_dict, max_dict = {}, {}

    def hook(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 4:  # Conv output
                mins = output.amin(dim=(0,2,3))
                maxs = output.amax(dim=(0,2,3))
            elif output.dim() == 2:  # Linear output
                mins = output.amin(dim=0)
                maxs = output.amax(dim=0)
            else:
                raise ValueError("Unexpected tensor shape for profiling")
            min_dict[name] = mins
            max_dict[name] = maxs
        return fn

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            model(images)

    for h in handles:
        h.remove()

    return min_dict, max_dict

# =========================================================
# 5️⃣ Hardening ResNet50 (CONV + FC)
# =========================================================

def harden_model(model, vuln_dict, min_dict, max_dict, ratio=0.1):
    model = copy.deepcopy(model)

    last_fc_name = get_last_linear_name(model)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):

            # ✅ Paper rule: duplicate ALL neurons in last FC
            if name == last_fc_name:
                top_idx = list(range(module.out_features))
            else:
                scores = torch.tensor(vuln_dict[name]["scores"])
                k = max(1, int(len(scores) * ratio))
                top_idx = torch.topk(scores, k).indices.tolist()

            parent = get_parent(model, name)
            child = name.split('.')[-1]

            dup_layer = DuplicatedLayer(module, top_idx)
            edac_layer = EDACLayer(
                min_dict[name],
                max_dict[name],
                top_idx
            )
            hardened_block = HardenedBlock(dup_layer, edac_layer)

            setattr(parent, child, hardened_block)

    return model


# =========================================================
# Hardened model (pruning-aware)
# =========================================================
def harden_model_pruned(model, vuln_dict, min_dict, max_dict, ratio=0.1, pruned_idx_dict=None):
    """
    Harden a pruned CNN with selective duplication + EDAC.
    """
    model = copy.deepcopy(model)
    last_fc_name = get_last_linear_name(model)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):

            # Remaining channels after pruning
            remaining_idx = pruned_idx_dict[name] if pruned_idx_dict and name in pruned_idx_dict else None

            # Duplicate all neurons in last FC
            if name == last_fc_name:
                top_idx = remaining_idx if remaining_idx is not None else list(range(module.out_features))
            else:
                scores = torch.tensor(vuln_dict[name]["scores"])
                if remaining_idx is not None:
                    scores = scores[remaining_idx]
                k = max(1, int(len(scores)*ratio))
                top_idx_local = torch.topk(scores, k).indices.tolist()
                if remaining_idx is not None:
                    top_idx = [remaining_idx[i] for i in top_idx_local]
                else:
                    top_idx = top_idx_local

                if isinstance(module, nn.Conv2d):
                    max_dim = module.out_channels
                elif isinstance(module, nn.Linear):
                    max_dim = module.out_features
                else:
                    continue  # safety

                top_idx = [i for i in top_idx if i < max_dim]

            parent = get_parent(model, name)
            child = name.split('.')[-1]

            dup_layer = DuplicatedLayer(module, top_idx)
            edac_layer = EDACLayer(min_dict[name], max_dict[name], top_idx)
            hardened_block = HardenedBlock(dup_layer, edac_layer)

            setattr(parent, child, hardened_block)

    return model
