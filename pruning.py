import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torchvision.models.resnet import BasicBlock, Bottleneck

# ================= Utility =================
def get_parent(model, name):
    components = name.split(".")
    parent = model
    for c in components[:-1]:
        parent = getattr(parent, c)
    return parent

def prune_bn(bn_layer, keep_idx):
    new_bn = nn.BatchNorm2d(len(keep_idx))
    new_bn.weight.data = bn_layer.weight.data[keep_idx].clone()
    new_bn.bias.data = bn_layer.bias.data[keep_idx].clone()
    new_bn.running_mean = bn_layer.running_mean[keep_idx].clone()
    new_bn.running_var = bn_layer.running_var[keep_idx].clone()
    new_bn.eps = bn_layer.eps
    new_bn.momentum = bn_layer.momentum
    return new_bn

def prune_conv(conv_layer, keep_out_idx, keep_in_idx=None):
    if keep_in_idx is None:
        keep_in_idx = list(range(conv_layer.in_channels))
    new_conv = nn.Conv2d(
        in_channels=len(keep_in_idx),
        out_channels=len(keep_out_idx),
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=(conv_layer.bias is not None)
    )
    new_conv.weight.data = conv_layer.weight.data[keep_out_idx][:, keep_in_idx].clone()
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data[keep_out_idx].clone()
    return new_conv

def prune_bn_for_conv(model, conv_name, keep_idx):
    modules = dict(model.named_modules())
    candidate = conv_name.replace("conv", "bn")
    if candidate in modules and isinstance(modules[candidate], nn.BatchNorm2d):
        bn = modules[candidate]
        keep_idx = [i for i in keep_idx if i < bn.num_features]
        if len(keep_idx) == 0:
            return
        parent = get_parent(model, candidate)
        child_name = candidate.split('.')[-1]
        new_bn = nn.BatchNorm2d(len(keep_idx))
        new_bn.weight.data = bn.weight.data[keep_idx].clone()
        new_bn.bias.data = bn.bias.data[keep_idx].clone()
        new_bn.running_mean = bn.running_mean[keep_idx].clone()
        new_bn.running_var = bn.running_var[keep_idx].clone()
        new_bn.eps = bn.eps
        new_bn.momentum = bn.momentum
        setattr(parent, child_name, new_bn)
        
# ================= Pruning Function =================
def pruning_model(model, vuln_dict, conv_prune_ratios, fc_prune_ratios):
    model = copy.deepcopy(model)
    pruned_idx_dict = {}
    conv_idx, fc_idx = 0, 0
    is_resnet = any(isinstance(m, (BasicBlock, Bottleneck)) for m in model.modules())

    if is_resnet:
        prev_keep_idx = list(range(model.conv1.out_channels))
        for layer_name in ["layer1","layer2","layer3","layer4"]:
            layer = getattr(model, layer_name)
            for block_idx, block in enumerate(layer):
                block_prefix = f"{layer_name}.{block_idx}"

                # conv1
                conv_name = block_prefix + ".conv1"
                scores = torch.tensor(vuln_dict[conv_name]["scores"])
                keep_out_idx1 = torch.topk(scores, max(1,int(len(scores)*(1-conv_prune_ratios[0])))).indices.tolist()
                block.conv1 = prune_conv(block.conv1, keep_out_idx1, prev_keep_idx)
                block.bn1 = prune_bn(block.bn1, keep_out_idx1)

                # conv2
                conv_name = block_prefix + ".conv2"
                scores = torch.tensor(vuln_dict[conv_name]["scores"])
                keep_out_idx2 = torch.topk(scores, max(1,int(len(scores)*(1-conv_prune_ratios[1])))).indices.tolist()
                block.conv2 = prune_conv(block.conv2, keep_out_idx2, keep_out_idx1)
                block.bn2 = prune_bn(block.bn2, keep_out_idx2)

                # conv3
                conv_name = block_prefix + ".conv3"
                scores = torch.tensor(vuln_dict[conv_name]["scores"])
                keep_out_idx3 = torch.topk(scores, max(1,int(len(scores)*(1-conv_prune_ratios[2])))).indices.tolist()
                block.conv3 = prune_conv(block.conv3, keep_out_idx3, keep_out_idx2)
                block.bn3 = prune_bn(block.bn3, keep_out_idx3)

                # downsample
                if block.downsample is not None:
                    ds_conv = block.downsample[0]
                    ds_bn = block.downsample[1]
                    block.downsample[0] = prune_conv(ds_conv, keep_out_idx3, prev_keep_idx)
                    block.downsample[1] = prune_bn(ds_bn, keep_out_idx3)

                prev_keep_idx = keep_out_idx3
                pruned_idx_dict[block_prefix] = keep_out_idx3

        # final FC
        old_fc = model.fc
        model.fc = nn.Linear(len(prev_keep_idx), old_fc.out_features)
        return model, pruned_idx_dict

    else:
        # AlexNet / VGG
        prev_keep_idx = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                scores = torch.tensor(vuln_dict[name]["scores"])
                prune_ratio = conv_prune_ratios[min(conv_idx, len(conv_prune_ratios)-1)]
                keep_k = max(1, int(len(scores)*(1-prune_ratio)))
                keep_idx = torch.topk(scores, keep_k).indices.tolist()
                conv_idx += 1

                parent = get_parent(model, name)
                child_name = name.split('.')[-1]

                if prev_keep_idx is None:
                    keep_in_idx = None
                else:
                    keep_in_idx = [i for i in prev_keep_idx if i < module.in_channels]
                    if len(keep_in_idx)==0: keep_in_idx = list(range(module.in_channels))

                new_conv = prune_conv(module, keep_idx, keep_in_idx)
                setattr(parent, child_name, new_conv)
                prune_bn_for_conv(model, name, keep_idx)
                prev_keep_idx = keep_idx
                pruned_idx_dict[name] = keep_idx

            elif isinstance(module, nn.Linear):
                scores = torch.tensor(vuln_dict[name]["scores"])
                prune_ratio = fc_prune_ratios[min(fc_idx, len(fc_prune_ratios)-1)]
                keep_k = max(1, int(len(scores)*(1-prune_ratio)))
                keep_idx = torch.topk(scores, keep_k).indices.tolist()
                fc_idx += 1

                parent = get_parent(model, name)
                child_name = name.split('.')[-1]

                in_features = module.in_features if prev_keep_idx is None else len(prev_keep_idx)
                new_fc = nn.Linear(in_features, len(keep_idx), bias=(module.bias is not None))
                if prev_keep_idx is None:
                    new_fc.weight.data = module.weight.data[keep_idx].clone()
                else:
                    new_fc.weight.data = module.weight.data[keep_idx][:, prev_keep_idx].clone()
                if module.bias is not None:
                    new_fc.bias.data = module.bias.data[keep_idx].clone()
                setattr(parent, child_name, new_fc)
                pruned_idx_dict[name] = keep_idx

        return model, pruned_idx_dict

# ================= Lightweight Retraining =================
def lightweight_retraining(model, train_loader, device='cuda', epochs=10, lr=0.001):
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model