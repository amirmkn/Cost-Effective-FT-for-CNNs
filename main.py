import torch
import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import copy
import csv
import matplotlib.pyplot as plt
import os
from vulnerability import VulnerabilityAnalyzer
from duplication import select_top_channels, apply_duplication
from fault_injection import inject_bitflips
from evaluation import evaluate
from edac import EDACLayer
from model import load_resnet50_from_pth

# Set device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

datasets_list = ["cifar10", "cifar100"]
# datasets_list = ["imagenet"]
bers = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
n_runs = 2

def get_dataset(name, max_samples):
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    ])
    if name.lower() == "cifar10":
        ds_full = dsets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    elif name.lower() == "cifar100":
        ds_full = dsets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        num_classes = 100
    elif name.lower() == "imagenet":
        ds_full = dsets.ImageNet(root="./data/imagenet", split='val', download=False, transform=transform)
        num_classes = 1000
    else:
        raise ValueError("Unknown dataset")
    ds = torch.utils.data.Subset(ds_full, range(min(max_samples, len(ds_full))))
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    return loader, num_classes

os.makedirs("results", exist_ok=True)

for dname in datasets_list:

    print("\n===== DATASET:", dname, "=====")

    loader, num_classes = get_dataset(dname, max_samples=100)

    pth_files = {
    "imagenet": "./resnet50.pth",
    "cifar10": "./resnet50_cifar10.pth",      # اگر داشتی
    "cifar100": "./resnet50_cifar100.pth"     # اگر داشتی
    }

    model = load_resnet50_from_pth(dname, pth_files).to(device)

    # model = resnet50()

    # if dname == "cifar10":
    #     model.fc = torch.nn.Linear(model.fc.in_features, 10)
    #     model.load_state_dict(torch.load("resnet50_cifar10.pth"))

    # elif dname == "cifar100":
    #     model.fc = torch.nn.Linear(model.fc.in_features, 100)
    #     model.load_state_dict(torch.load("resnet50_cifar100.pth"))

    # elif dname == "imagenet":
    #     model.load_state_dict(torch.load("resnet50.pth"))

    # model = model.to(device)

    # ---------------- Baseline ----------------
    print("-1")
    base_top1, base_top5 = evaluate(model, loader, device)
    print("Base Top-1: ", base_top1, "Top-5: ",base_top5)
    print("0")

    # ---------------- Vulnerability + Hardening ----------------
    analyzer = VulnerabilityAnalyzer(model, device)
    print("1")
    vuln = analyzer.analyze(loader, max_batches=1)
    print("2")


    selected = select_top_channels(vuln, 0.1)
    print("3")

    total = sum(p.abs().sum().item() for p in model.parameters())
    print("Total weight magnitude:", total)

    hardened = apply_duplication(model, selected)
    print("4")

    hardened.to(device)
    print("5")

    # ---------------- EDAC ----------------
    print("6")

    for name, module in hardened.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            num_ch = module.out_channels if isinstance(module, torch.nn.Conv2d) else module.out_features
            dup_idx = selected.get(name, []) 
            parent = hardened
            names = name.split('.')
            for n in names[:-1]:
                parent = getattr(parent, n)
            setattr(parent, names[-1], torch.nn.Sequential(module, EDACLayer(num_ch, duplicated_idx=dup_idx)))

    hardened.to(device)
    print("7")

    # ---------------- CSV ----------------
    csv_file = f"results/{dname}_BER_results.csv"

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["BER", "Run", "Top-1", "Top-5"])

        ber_means = []
        ber_stds = []

        for ber in bers:

            accs = []

            for run in range(n_runs):

                m = copy.deepcopy(hardened)
                inject_bitflips(m, ber)

                top1, top5 = evaluate(m, loader, device)

                accs.append(top1)
                writer.writerow([ber, run+1, top1, top5])

                print(f"{dname} | BER={ber} | Run={run+1} | Top1={top1:.4f}")

            mean = sum(accs)/len(accs)
            std = (sum([(a-mean)**2 for a in accs])/len(accs))**0.5

            ber_means.append(mean)
            ber_stds.append(std)

            print(f"BER={ber}: mean={mean:.4f}, std={std:.4f}")

    # -------------- plot -----------------
    plt.figure()
    plt.errorbar(bers, ber_means, yerr=ber_stds, fmt='-o', capsize=5)
    plt.xscale('log')
    plt.xlabel("Bit Error Rate (BER)")
    plt.ylabel("Top-1 Accuracy")
    plt.title(f"ResNet50 Hardened - {dname}")
    plt.grid(True)
    plt.savefig(f"results/{dname}_BER_plot.png")
    plt.show()
