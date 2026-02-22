import torch
import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import copy
import csv
import matplotlib.pyplot as plt
import os
import time
from vulnerability import VulnerabilityAnalyzer
from duplication import select_top_channels, apply_duplication
from fault_injection import inject_bitflips
from evaluation import evaluate
from edac import EDACLayer
from model import load_resnet50_from_pth, load_alexnet, load_vgg11, load_vgg16
from hardening import harden_model, profile_model

# ================= Model selection =================
MODEL_NAME = "resnet50"
# MODEL_NAME = "alexnet"
# MODEL_NAME = "vgg11"
# MODEL_NAME = "vgg16"

# Set device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# datasets_list = ["cifar10", "cifar100"]
# datasets_list = ["imagenet"]
if MODEL_NAME == "alexnet":
    datasets_list = ["cifar10"]
elif MODEL_NAME == "vgg11":
    datasets_list = ["cifar10"]
elif MODEL_NAME == "vgg16":
    datasets_list = ["cifar100"]
elif MODEL_NAME == "resnet50":
    datasets_list = ["cifar10", "cifar100"]
else:
    raise ValueError("Unknown MODEL_NAME")

bers = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
n_runs = 15
Harden_ratio = 0.15

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

    loader, num_classes = get_dataset(dname, max_samples=500)

    # model = load_resnet50_from_pth("./resnet50.pth", num_classes=num_classes).to(device)
    
    # pth_files = {
    # "imagenet": "./resnet50.pth",
    # "cifar10": "./resnet50_cifar10.pth", 
    # "cifar100": "./resnet50_cifar100.pth"
    # }
 # ================= Load model =================
    if MODEL_NAME == "alexnet":
        model = load_alexnet(
            num_classes=10,
            pth_path="./alexnet_cifar10.pth"
        ).to(device)

    elif MODEL_NAME == "vgg11":
        model = load_vgg11(
            num_classes=10,
            pth_path="./vgg11_cifar10.pth"
        ).to(device)

    elif MODEL_NAME == "vgg16":
        model = load_vgg16(
            num_classes=100,
            pth_path="./vgg16_cifar100.pth"
        ).to(device)

    elif MODEL_NAME == "resnet50":
        pth_files = {
            "imagenet": "./resnet50.pth",
            "cifar10": "./resnet50_cifar10.pth",
            "cifar100": "./resnet50_cifar100.pth"
        }
        model = load_resnet50_from_pth(dname, pth_files).to(device)

    else:
        raise ValueError("Invalid MODEL_NAME")

    model = load_resnet50_from_pth(dname, pth_files).to(device)

    # ---------------- Baseline ----------------
    base_acc, base_top5, base_top10 = evaluate(model, loader, device)
    print("base_acc = ",base_acc,"base_top5 = ", base_top5,"base_top10 = ", base_top10)

    # ---------------- Vulnerability + Hardening ----------------
    analyzer = VulnerabilityAnalyzer(model, device)

    vuln = analyzer.analyze(loader, max_batches=1)

    selected = select_top_channels(vuln, Harden_ratio)

    min_dict, max_dict = profile_model(model, loader, device)

    # Hardening
    hardened_model = harden_model(
        model, vuln, min_dict, max_dict, ratio=Harden_ratio
    ).to(device)

    # ---------------- CSV ----------------
    csv_file = f"results/{dname}_BER_results.csv"

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["BER", "Run", "Accuracy", "AccuracyDrop(%)", "Top-5", "Top-10"])

        top5_means, top5_stds = [], []
        top10_means, top10_stds = [], []
        acc_means, acc_stds = [], []

        for ber in bers:

            top5_list = []
            top10_list = []
            acc_list = []
            drop_percent_list = []

            for run in range(n_runs):

                m = copy.deepcopy(hardened_model)
                inject_bitflips(m, ber)

                acc ,top5, top10 = evaluate(m, loader, device)
                drop_percent = 100 * (base_acc - acc)

                acc_list.append(acc)
                top5_list.append(top5)
                top10_list.append(top10)
                drop_percent_list.append(drop_percent)

                writer.writerow([ber, run+1, acc, drop_percent, top5, top10])

                print(f"{dname} | BER={ber} | Run={run+1} | "
                    f"Acc={acc:.4f} | Top5={top5:.4f} | "
                    f"Top10={top10:.4f} ")

            # ---  mean و std ---
            def mean_std(lst):
                mean = sum(lst)/len(lst)
                std = (sum([(a-mean)**2 for a in lst])/len(lst))**0.5
                return mean, std
            
            ma, sa = mean_std(acc_list)
            m5, s5 = mean_std(top5_list)
            m10, s10 = mean_std(top10_list)

            acc_means.append(ma); acc_stds.append(sa)
            top5_means.append(m5); top5_stds.append(s5)
            top10_means.append(m10); top10_stds.append(s10)

            print(f"BER={ber}: "
                f"Acc mean={ma:.4f}, "
                f"Top5 mean={m5:.4f}, "
                f"Top10 mean={m10:.4f}, "
                )

    # ------------------ Plot  ------------------
    def plot_metric(means, stds, metric_name):
        plt.figure()
        plt.errorbar(bers, means, yerr=stds, fmt='-o', capsize=5)
        plt.xscale('log')
        plt.xlabel("Bit Error Rate (BER)")
        plt.ylabel(metric_name)
        plt.title(f"ResNet50 Hardened - {dname} ({metric_name})")
        plt.grid(True)
        plt.savefig(f"results/{dname}_{metric_name}_plot.png")
        plt.close()


    plot_metric(top5_means, top5_stds, "Top5")
    plot_metric(top10_means, top10_stds, "Top10")
    plot_metric(acc_means, acc_stds, "Accuracy")
    plot_metric(acc_means, acc_stds, "AccuracyDrop(%)")


def measure_time(model, loader):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
    return time.time() - start

baseline_time = measure_time(model, loader)
hardened_time = measure_time(hardened_model, loader)

overhead_percent = 100 * (hardened_time - baseline_time) / baseline_time
print("Performance overhead (%):", overhead_percent)
