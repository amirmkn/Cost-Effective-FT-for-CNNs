import torch
import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import copy
import csv
import os
import time
from vulnerability import VulnerabilityAnalyzer
from fault_injection import inject_bitflips
from evaluation import evaluate
from model import load_resnet50, load_alexnet, load_vgg11, load_vgg16
from hardening import harden_model, profile_model , harden_model_pruned 
from pruning import pruning_model,lightweight_retraining, rebuild_first_fc
from plotting import plot_single_model_metric, plot_baseline_vs_pruned, plot_pareto_overhead_vs_accuracy

def main():
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
        datasets_list = ["cifar10", "cifar100","tiny_imagenet"]
    else:
        raise ValueError("Unknown MODEL_NAME")

    # bers = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4] # BERs to test
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
        elif name.lower() == "tiny_imagenet":
            ds_full = dsets.ImageNet(root="./data/tiny-imagenet-200", split='val', download=False, transform=transform)
            num_classes = 200
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

        loader, num_classes = get_dataset(dname, max_samples = None)

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
                "imagenet": "./resnet50_tiny_best.pth",
                "cifar10": "./resnet50_cifar10.pth",
                "cifar100": "./resnet50_cifar100.pth"
            }
            model = load_resnet50(dname, pth_files).to(device)

        else:
            raise ValueError("Invalid MODEL_NAME")


        # Baseline
        base_acc, base_top5, base_top10, _, _, _ = evaluate(model, loader, device)
        print("base_acc = ",base_acc,"base_top5 = ", base_top5,"base_top10 = ", base_top10)

        # Vulnerability + Hardening 
        print("start analyzing...")
        analyzer = VulnerabilityAnalyzer(model, device)

        vuln = analyzer.analyze(loader, max_batches=30)
        # ===== Profiling original model =====
        min_dict, max_dict = profile_model(model, loader, device)

        print("start calculating Hardened Baseline")
        hardened_baseline = harden_model(
            model,
            vuln,
            min_dict,
            max_dict,
            ratio=Harden_ratio
        ).to(device)

        print ("start pruning...")
        pruning_ratios_dict = {
            # AlexNet
            ("alexnet", "cifar10"): {
                "conv": [0.05]*5,   # 5 لایه کانولوشن
                "fc":   [0.8]*3     # 3 لایه FC
            },
            # VGG-11
            ("vgg11", "cifar10"): {
                "conv": [0.04]*8,   # 8 لایه کانولوشن
                "fc":   [0.35]*3    # 3 لایه FC
            },
            # VGG-16
            ("vgg16", "cifar100"): {
                "conv": [0.01]*13,  # 13 لایه کانولوشن
                "fc":   [0.15]*3    # 3 لایه FC
            },
            # ResNet-50
            ("resnet50", "cifar10"): {
                "conv": [0.05]*16,  # 16 لایه کانولوشن اصلی
                "fc":   [0.8]       # 1 لایه FC آخر
            },
            ("resnet50", "tiny_imagenet"): {
                "conv": [0.02]*16,
                "fc":   [0.10]
            },
            ("resnet50", "cifar100"): {
                "conv": [0.04]*16,  # 16 لایه کانولوشن اصلی
                "fc":   [0.35]      # 1 لایه FC آخر
            }
        }
        if (MODEL_NAME, dname) in pruning_ratios_dict:
            conv_prune_ratios = pruning_ratios_dict[(MODEL_NAME, dname)]["conv"]
            fc_prune_ratios   = pruning_ratios_dict[(MODEL_NAME, dname)]["fc"]

        pruned_model , pruned_idx_dict = pruning_model(model, vuln, conv_prune_ratios, fc_prune_ratios)
        if MODEL_NAME in ["alexnet", "vgg11", "vgg16"]:
            pruned_model = rebuild_first_fc(pruned_model, input_size=(3,32,32), device=device)
        print("start retraining...")
        pruned_model = lightweight_retraining(pruned_model, loader, device, epochs=1)
        min_dict, max_dict = profile_model(pruned_model, loader, device)

        print("start hardening...")
        # Hardening
        hardened_model = harden_model_pruned(pruned_model, vuln, min_dict, max_dict, ratio=Harden_ratio, pruned_idx_dict=pruned_idx_dict).to(device)
        
        # Measure performance overhead
        print("\n=== Measuring Performance Overhead ===")
        baseline_time = measure_time(model, loader, device)
        hardened_base_time = measure_time(hardened_baseline, loader, device)
        hardened_pruned_time = measure_time(hardened_model, loader, device)
        
        overhead_base = 100 * (hardened_base_time - baseline_time) / baseline_time
        overhead_pruned = 100 * (hardened_pruned_time - baseline_time) / baseline_time

        print(f"Baseline inference time:  {baseline_time:.2f} sec")
        print(f"Hardened baseline inference time (no pruning):  {hardened_base_time:.2f} sec")
        print(f"Hardened pruned inference time:   {hardened_pruned_time:.2f} sec")
        print(f"Overhead baseline (%):            {overhead_base:.2f}%")
        print(f"Overhead pruned (%):              {overhead_pruned:.2f}%")
        
        # CSV logging 
        csv_file = f"results/{MODEL_NAME}_{dname}_BER_results.csv"

        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["BER", "Run", "Accuracy(Haedened+Pruned)", "Accuracy Drop (%)", "Top5", "Top10", "Precision", "Recall", "F1 Score"])
            f.flush() #flush header immediately
            writer.writerow([])
            writer.writerow(["Overhead Baseline (%)", overhead_base])
            writer.writerow(["Overhead Pruned (%)", overhead_pruned])
            writer.writerow(["Baseline Time (s)", baseline_time])
            writer.writerow(["Hardened Baseline Time (s)", hardened_base_time])
            writer.writerow(["Hardened Time (s)", hardened_pruned_time])
            writer.writerow([])
            f.flush()
            
            # Containers for plotting
            top5_means, top5_stds = [], []
            top10_means, top10_stds = [], []
            acc_means, acc_stds = [], []
            drop_means, drop_stds = [], []
            drop_baseline_means, drop_pruned_means = [], []

            for ber in bers:
                drops_baseline = []
                drops_pruned = []
                top5_list = []
                top10_list = []
                acc_list = []
                drop_percent_list = []

                for run in range(n_runs):
                    #Hardened baseline
                    m_base = copy.deepcopy(hardened_baseline)
                    inject_bitflips(m_base, ber)
                    accuracy_base, _, _, _, _, _ = evaluate(m_base, loader, device)
                    drop_base = 100 * (base_acc - accuracy_base)
                    drops_baseline.append(drop_base)
                    #Hardened pruned
                    m_pruned = copy.deepcopy(hardened_model)
                    inject_bitflips(m_pruned, ber)
                    accuracy_pruned, top5, top10, precision, recall, f1 = evaluate(m_pruned, loader, device)
                    drop_pruned = 100 * (base_acc - accuracy_pruned)
                    drops_pruned.append(drop_pruned)
                    acc_list.append(accuracy_pruned)
                    top5_list.append(top5)
                    top10_list.append(top10)
                    drop_percent_list.append(drop_pruned)

                    writer.writerow([ber, run+1, accuracy_pruned, drop_pruned, top5, top10, precision, recall, f1])

                    print(f"{dname} | BER={ber} | Run={run+1} | "
                        f"Acc={accuracy_pruned:.4f} | Top5={top5:.4f} | "
                        f"Top10={top10:.4f} | F1={f1:.4f} | Drop={drop_pruned:.2f}%")

                # ---- store means for plotting ----
                drop_baseline_means.append(sum(drops_baseline) / len(drops_baseline))
                drop_pruned_means.append(sum(drops_pruned) / len(drops_pruned))
                
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
    plot_baseline_vs_pruned(
                bers,
                drop_baseline_means,
                drop_pruned_means,
                MODEL_NAME,
                dname
            )

    plot_single_model_metric(bers, top5_means, top5_stds, "Top5", MODEL_NAME, dname)
    plot_single_model_metric(bers, top10_means, top10_stds, "Top10", MODEL_NAME, dname)
    plot_single_model_metric(bers, acc_means, acc_stds, "Accuracy", MODEL_NAME, dname)
    plot_single_model_metric(bers, drop_means, drop_stds, "AccuracyDrop(%)", MODEL_NAME, dname)
    plot_pareto_overhead_vs_accuracy(
        accuracy_drops=[
            sum(drop_baseline_means) / len(drop_baseline_means),
            sum(drop_pruned_means) / len(drop_pruned_means)
            ],
        overheads=[overhead_base, overhead_pruned],
        labels=["Hardened baseline", "Hardened pruned"],
        title=f"{MODEL_NAME} Pareto Trade-off"
)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()