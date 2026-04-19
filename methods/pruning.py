import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torchvision.models.resnet import BasicBlock, Bottleneck

def rebuild_first_fc(model, input_size=(3, 32, 32), device=None):
    # auto device
    if device is None:
        device = next(model.parameters()).device
    if str(device) == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model.eval()
    model = model.to(device)

    # 1) get new conv flattened size
    with torch.no_grad():
        dummy = torch.zeros(1, *input_size).to(device)
        flat_dim = model.features(dummy).view(1, -1).size(1)

    # 2) find the FIRST Linear layer inside classifier
    first_fc_name = None
    for name, module in model.classifier.named_modules():
        if isinstance(module, nn.Linear):
            first_fc_name = name
            break

    if first_fc_name is None:
        raise RuntimeError("No Linear layer found in model.classifier!")

    # NOTE: name may be nested (e.g. '1' or 'block.0')
    parent = model.classifier
    for part in first_fc_name.split('.')[:-1]:
        parent = parent[int(part)]
    key = first_fc_name.split('.')[-1]
    if key.isdigit():
        key = int(key)

    old_fc = parent[key]

    # 3) rebuild FC to match new input size
    new_fc = nn.Linear(flat_dim, old_fc.out_features).to(device)

    parent[key] = new_fc
    return model

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

def prune_linear(fc_layer, keep_out_idx, keep_in_idx=None):
    if keep_in_idx is None:
        keep_in_idx = list(range(fc_layer.in_features))

    new_fc = nn.Linear(
        in_features=len(keep_in_idx),
        out_features=len(keep_out_idx),
        bias=(fc_layer.bias is not None)
    )

    new_fc.weight.data = fc_layer.weight.data[keep_out_idx][:, keep_in_idx].clone()

    if fc_layer.bias is not None:
        new_fc.bias.data = fc_layer.bias.data[keep_out_idx].clone()

    return new_fc
# Pruning Function
def pruning_model(model, vuln_dict, conv_prune_ratios, fc_prune_ratios, device):
    model = copy.deepcopy(model)
    pruned_idx_dict = {}
    conv_idx, fc_idx = 0, 0
    prev_keep_idx = None
    is_resnet = any(isinstance(m, (BasicBlock, Bottleneck)) for m in model.modules())

    if is_resnet:
        print("\n" + "="*60)
        print("Starting ResNet model pruning")
        print("="*60)

        # ========== 1) Prune first conv and bn ==========
        if hasattr(model, 'conv1') and isinstance(model.conv1, nn.Conv2d):
            print("\n[STEP 1] Pruning conv1 and bn1")
            old_out_channels = model.conv1.out_channels
            old_in_channels = model.conv1.in_channels
            print(f"  Before pruning - conv1: in={old_in_channels}, out={old_out_channels}")

            scores = torch.tensor(vuln_dict['conv1']["scores"])
            prune_ratio = conv_prune_ratios[0] if conv_prune_ratios else 0.5
            keep_k = max(1, int(len(scores) * (1 - prune_ratio)))
            keep_idx_conv1 = torch.topk(scores, keep_k).indices.tolist()
            print(f"  Prune ratio: {prune_ratio:.2f} -> kept channels: {len(keep_idx_conv1)} out of {old_out_channels}")

            model.conv1 = prune_conv(model.conv1, keep_idx_conv1, None)
            print(f"  After pruning - conv1: in={model.conv1.in_channels}, out={model.conv1.out_channels}")

            if hasattr(model, 'bn1') and isinstance(model.bn1, nn.BatchNorm2d):
                old_bn_features = model.bn1.num_features
                model.bn1 = prune_bn(model.bn1, keep_idx_conv1)
                print(f"  After pruning - bn1: num_features = {model.bn1.num_features} (was {old_bn_features})")

            prev_keep_idx = keep_idx_conv1
            pruned_idx_dict['conv1'] = keep_idx_conv1

        # ========== 2) Prune residual layers ==========
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            print(f"\n[LAYER] {layer_name}")
            layer = getattr(model, layer_name)
            for block_idx, block in enumerate(layer):
                block_prefix = f"{layer_name}.{block_idx}"
                is_bottleneck = isinstance(block, Bottleneck)
                print(f"\n  Block {block_prefix} ({'Bottleneck' if is_bottleneck else 'BasicBlock'})")
                print(f"    Block input (prev_keep_idx) = {len(prev_keep_idx)} channels")

                # -------------- conv1 --------------
                conv_name = block_prefix + ".conv1"
                print(f"\n    --- conv1 ({conv_name}) ---")
                old_conv = block.conv1
                print(f"      Before pruning: in={old_conv.in_channels}, out={old_conv.out_channels}")

                scores = torch.tensor(vuln_dict[conv_name]["scores"])
                keep_out_idx1 = torch.topk(scores, max(1, int(len(scores)*(1-conv_prune_ratios[0])))).indices.tolist()
                print(f"      Prune ratio = {conv_prune_ratios[0]:.2f} -> keeping {len(keep_out_idx1)} out of {len(scores)} output channels")

                block.conv1 = prune_conv(block.conv1, keep_out_idx1, prev_keep_idx)
                print(f"      After pruning: in={block.conv1.in_channels}, out={block.conv1.out_channels}")

                block.bn1 = prune_bn(block.bn1, keep_out_idx1)
                print(f"      bn1 pruned: num_features = {block.bn1.num_features}")

                # -------------- conv2 --------------
                conv_name = block_prefix + ".conv2"
                print(f"\n    --- conv2 ({conv_name}) ---")
                old_conv = block.conv2
                print(f"      Before pruning: in={old_conv.in_channels}, out={old_conv.out_channels}")

                scores = torch.tensor(vuln_dict[conv_name]["scores"])
                keep_out_idx2 = torch.topk(scores, max(1, int(len(scores)*(1-conv_prune_ratios[1])))).indices.tolist()
                print(f"      Prune ratio = {conv_prune_ratios[1]:.2f} -> keeping {len(keep_out_idx2)} out of {len(scores)} output channels")

                block.conv2 = prune_conv(block.conv2, keep_out_idx2, keep_out_idx1)
                print(f"      After pruning: in={block.conv2.in_channels}, out={block.conv2.out_channels}")

                block.bn2 = prune_bn(block.bn2, keep_out_idx2)
                print(f"      bn2 pruned: num_features = {block.bn2.num_features}")

                # -------------- conv3 (only for Bottleneck) --------------
                if is_bottleneck:
                    conv_name = block_prefix + ".conv3"
                    print(f"\n    --- conv3 ({conv_name}) ---")
                    old_conv = block.conv3
                    print(f"      Before pruning: in={old_conv.in_channels}, out={old_conv.out_channels}")

                    scores = torch.tensor(vuln_dict[conv_name]["scores"])
                    has_downsample = (block.downsample is not None)
                    if has_downsample:
                        keep_k = max(1, int(len(scores)*(1-conv_prune_ratios[2])))
                        print(f"      Has downsample -> prune ratio = {conv_prune_ratios[2]:.2f}")
                    else:
                        keep_k = len(prev_keep_idx)
                        if keep_k > len(scores):
                            keep_k = len(scores)
                        print(f"      No downsample -> output channels must match block input: {keep_k} channels")

                    keep_out_idx3 = torch.topk(scores, keep_k).indices.tolist()
                    print(f"      Keeping {len(keep_out_idx3)} out of {len(scores)} output channels")

                    block.conv3 = prune_conv(block.conv3, keep_out_idx3, keep_out_idx2)
                    print(f"      After pruning: in={block.conv3.in_channels}, out={block.conv3.out_channels}")

                    block.bn3 = prune_bn(block.bn3, keep_out_idx3)
                    print(f"      bn3 pruned: num_features = {block.bn3.num_features}")
                else:
                    # BasicBlock: conv2 is the final output
                    keep_out_idx3 = keep_out_idx2
                    print(f"\n    (BasicBlock - conv2 used as final output)")

                # -------------- downsample --------------
                if block.downsample is not None:
                    print(f"\n    --- downsample ---")
                    ds_conv = block.downsample[0]
                    ds_bn = block.downsample[1]
                    print(f"      Before pruning - downsample conv: in={ds_conv.in_channels}, out={ds_conv.out_channels}")
                    print(f"      Before pruning - downsample bn: features={ds_bn.num_features}")

                    block.downsample[0] = prune_conv(ds_conv, keep_out_idx3, prev_keep_idx)
                    block.downsample[1] = prune_bn(ds_bn, keep_out_idx3)

                    print(f"      After pruning - downsample conv: in={block.downsample[0].in_channels}, out={block.downsample[0].out_channels}")
                    print(f"      After pruning - downsample bn: features={block.downsample[1].num_features}")

                prev_keep_idx = keep_out_idx3
                pruned_idx_dict[block_prefix] = keep_out_idx3
                print(f"    Block output: {len(prev_keep_idx)} channels (will be input to next block)")

        # ========== 3) Prune final fully connected layer ==========
        print("\n[STEP 3] Pruning final fully connected layer (fc)")
        old_fc = model.fc
        print(f"  Before pruning - fc: in_features={old_fc.in_features}, out_features={old_fc.out_features}")
        new_fc = nn.Linear(len(prev_keep_idx), old_fc.out_features)
        new_fc.weight.data = old_fc.weight.data[:, prev_keep_idx].clone()
        new_fc.bias.data = old_fc.bias.data.clone()
        model.fc = new_fc
        print(f"  After pruning - fc: in_features={model.fc.in_features}, out_features={model.fc.out_features}")

        print("\n" + "="*60)
        print("ResNet pruning completed successfully")
        print("="*60 + "\n")

        return model, pruned_idx_dict

    else:
        # AlexNet / VGG
        prev_keep_idx = None
        for name, module in model.named_modules():
            #CONV PRUNING
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
            # 2) REBUILD FIRST FC TO MATCH NEW CONV OUTPUT SIZE
        model = rebuild_first_fc(model, input_size=(3,32,32), device=device)

        # ----- FC PRUNING -----
        prev_keep_idx = None
        fc_layers = list(model.classifier.named_modules())

        for name, module in model.classifier.named_modules():
            # Only direct children (avoid nested)
            if not isinstance(module, nn.Linear):
                continue

            # detect last FC
            is_last_fc = (module is model.classifier[-1])

            if is_last_fc:
                # Do NOT prune last FC output, but DO prune input later
                continue

            scores = torch.tensor(vuln_dict[f"classifier.{name}"]["scores"])
            prune_ratio = fc_prune_ratios[min(fc_idx, len(fc_prune_ratios)-1)]

            keep_k = max(1, int(len(scores)*(1-prune_ratio)))
            keep_idx = torch.topk(scores, keep_k).indices.tolist()
            fc_idx += 1

            parent = model.classifier
            key = int(name)

            if prev_keep_idx is None:
                keep_in_idx = list(range(module.in_features))
            else:
                keep_in_idx = [i for i in prev_keep_idx if i < module.in_features]

            new_fc = prune_linear(module, keep_idx, keep_in_idx)
            parent[key] = new_fc

            prev_keep_idx = keep_idx

        # FIX FINAL FC
        last_fc = model.classifier[-1]
        correct_in = len(prev_keep_idx)
        correct_out = last_fc.out_features

        new_last_fc = nn.Linear(correct_in, correct_out)
        new_last_fc.weight.data = last_fc.weight.data[:, prev_keep_idx].clone()
        new_last_fc.bias.data = last_fc.bias.data.clone()

        model.classifier[-1] = new_last_fc

        return model, pruned_idx_dict

# Lightweight Retraining
def lightweight_retraining(model, train_loader, device='cuda', epochs=10, lr=0.001):
    model = model.to(device)
    model.train()
    is_resnet = any(isinstance(m, (BasicBlock, Bottleneck)) for m in model.modules())
    if is_resnet:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
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