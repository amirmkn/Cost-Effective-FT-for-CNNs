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
from duplication import select_top_channels
from fault_injection import inject_bitflips
from evaluation import evaluate
from edac import EDACLayer
from model import load_resnet50_from_pth, load_alexnet, load_vgg11, load_vgg16
from hardening import harden_model, profile_model

# ================= Model selection =================
# MODEL_NAME = "resnet50"
MODEL_NAME = "alexnet"
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

bers = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4] # BERs to test
bers = [1e-6, 1e-5, 1e-4, 1e-3] # BERs Based on the IEEE paper
n_runs = 15
Harden_ratio = 0.15

def measure_time(model, loader, device):
    """Measure total inference time over one full pass of the dataloader."""
    model.eval()
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking = True)
            _ = model(x)
    torch.cuda.synchronize() if device == "cuda" else None
    return time.time() - start

def get_dataset(name, max_samples = None):
    # Set resize value based on the model being used
    # AlexNet (CIFAR version) needs 32. ResNet50/VGG typically use 224.
    img_size = 32 if MODEL_NAME == "alexnet" else 224
    transform = T.Compose([T.Resize(img_size),T.ToTensor(),
                           T.Normalize(
            mean=[0.4914, 0.4822, 0.4465], # Using CIFAR-10 stats
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    if name.lower() == "cifar10":
        transform = T.Compose([T.Resize(img_size),T.ToTensor(),
                           T.Normalize(
        mean=[0.4914, 0.4822, 0.4465], # Using CIFAR-10 stats
        std=[0.2023, 0.1994, 0.2010]
        )
    ])
        ds_full = dsets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    elif name.lower() == "cifar100":
        transform = T.Compose([T.Resize(img_size),T.ToTensor(), 
                           T.Normalize(
        mean=[0.5071, 0.4867, 0.4408], # Using CIFAR-100 stats
        std=[0.2675, 0.2565, 0.2761]
        )
        ])
        ds_full = dsets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        num_classes = 100
    elif name.lower() == "imagenet":
        ds_full = dsets.ImageNet(root="./data/imagenet", split='val', download=False, transform=transform)
        num_classes = 1000
    else:
        raise ValueError("Unknown dataset")
    
    # If max_samples is None or <= 0, use the full dataset. Otherwise, create a subset.
    if max_samples is None or max_samples <= 0:
        ds = ds_full              
    else:
        ds = torch.utils.data.Subset(
            ds_full,
            range(min(max_samples, len(ds_full)))
        )

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    return loader, num_classes

os.makedirs("results", exist_ok=True)

for dname in datasets_list:

    print("\n===== DATASET:", dname, "=====")

    loader, num_classes = get_dataset(dname, max_samples = 10)

 # Load model
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


    # Baseline
    base_acc, base_top5, base_top10 = evaluate(model, loader, device)
    print("base_acc = ",base_acc,"base_top5 = ", base_top5,"base_top10 = ", base_top10)

    # Vulnerability + Hardening
    analyzer = VulnerabilityAnalyzer(model, device)

    vuln = analyzer.analyze(loader, max_batches=30)

    selected = select_top_channels(vuln, Harden_ratio)

    min_dict, max_dict = profile_model(model, loader, device)

    # Hardening
    hardened_model = harden_model(model, vuln, min_dict, max_dict, ratio=Harden_ratio).to(device)
    
    # Measure performance overhead
    print("\n=== Measuring Performance Overhead ===")
    baseline_time = measure_time(model, loader, device)
    hardened_time = measure_time(hardened_model, loader, device)
    overhead_percent = 100 * (hardened_time - baseline_time) / baseline_time

    print(f"Baseline inference time:  {baseline_time:.2f} sec")
    print(f"Hardened inference time:  {hardened_time:.2f} sec")
    print(f"Performance overhead (%): {overhead_percent:.2f}%")
    
    # CSV logging 
    csv_file = f"results/{dname}_BER_results.csv"

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["BER", "Run", "Accuracy", "AccuracyDrop(%)", "Top-5", "Top-10"])
        f.flush() #flush header immediately
        writer.writerow([])
        writer.writerow(["Performance Overhead (%)", overhead_percent])
        writer.writerow(["Baseline Time (s)", baseline_time])
        writer.writerow(["Hardened Time (s)", hardened_time])
        writer.writerow([])
        f.flush()
        
        # Containers for plotting
        top5_means, top5_stds = [], []
        top10_means, top10_stds = [], []
        acc_means, acc_stds = [], []
        drop_means, drop_stds = [], []

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
            md, sd = mean_std(drop_percent_list)
            drop_means.append(md)
            drop_stds.append(sd)

            print(f"BER={ber}: "
                f"Acc mean={ma:.4f}, "
                f"Top5 mean={m5:.4f}, "
                f"Top10 mean={m10:.4f}, "
                )

    # Plotting
    def plot_metric(means, stds, metric_name):
        plt.figure()
        plt.errorbar(bers, means, yerr=stds, fmt='-o', capsize=5)
        plt.xscale('log')
        plt.xlabel("Bit Error Rate (BER)")
        plt.ylabel(metric_name)
        plt.title(f"{MODEL_NAME} Hardened - {dname} ({metric_name})")
        plt.grid(True)
        plt.savefig(f"results/{dname}_{metric_name}_plot.png")
        plt.close()


    plot_metric(top5_means, top5_stds, "Top5")
    plot_metric(top10_means, top10_stds, "Top10")
    plot_metric(acc_means, acc_stds, "Accuracy")
    plot_metric(drop_means, drop_stds, "AccuracyDrop(%)")


