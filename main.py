import os
import torch
from config import *
import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import copy
import csv
from models.model import load_resnet50, load_alexnet, load_vgg11, load_vgg16
from methods.vulnerability import VulnerabilityAnalyzer
from methods.fault_injection import inject_bitflips
from methods.hardening import harden_model, profile_model , harden_model_pruned 
from methods.pruning import pruning_model,lightweight_retraining, rebuild_first_fc
from utils.plotting import plot_single_model_metric, plot_baseline_vs_pruned, plot_pareto_overhead_vs_accuracy
from utils.evaluation import evaluate
from utils.timing import measure_time, get_timing_loader


def get_dataset(name, train = True, max_samples = None):
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
        ds_full = dsets.CIFAR10(root="./data", train=train, download=True, transform=transform)
        num_classes = 10
    elif name.lower() == "cifar100":
        transform = T.Compose([T.Resize(img_size),T.ToTensor(), 
                        T.Normalize(
        mean=[0.5071, 0.4867, 0.4408], # Using CIFAR-100 stats
        std=[0.2675, 0.2565, 0.2761]
        )
        ])
        ds_full = dsets.CIFAR100(root="./data", train=train, download=True, transform=transform)
        num_classes = 100
    elif name.lower() == "tiny_imagenet":
        split = "train" if train else "val"
        ds_full = dsets.ImageNet(root="./data/tiny-imagenet-200", split=split, download=False, transform=transform)
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

def load_model(model_name, dataset, device):
    # Load model
    if model_name == "alexnet":
        return load_alexnet(
            num_classes=10,
            pth_path=MODEL_WEIGHTS["alexnet"]
        ).to(device)

    elif model_name == "vgg11":
        return load_vgg11(
            num_classes=10,
            pth_path= MODEL_WEIGHTS["vgg11"]
            ).to(device)

    elif model_name == "vgg16":
        return load_vgg16(
            num_classes=100,
            pth_path=MODEL_WEIGHTS["vgg16"]
        ).to(device)

    elif model_name == "resnet50":

        if dataset == "cifar10":
            # path = MODEL_WEIGHTS["resnet50_cifar10"]
            path = "/kaggle/input/models/amirmkn/resnet50-cifar10/pytorch/default/1/resnet50_cifar10.pth"
            num_classes = 10

        if dataset == "cifar100":
            # path = MODEL_WEIGHTS["resnet50_cifar100"]
            path = "/kaggle/input/models/amirmkn/resnet50-cifar100/pytorch/default/1/resnet50_cifar100.pth"
            num_classes = 100

        if dataset == "tinyimagenet":
            # path = MODEL_WEIGHTS["resnet50_tiny"]
            path = "/kaggle/input/datasets/amirmkn/resnet-tiny-imagenet/resnet50_tiny_best.pth"
            num_classes = 200

        return load_resnet50(
            num_classes=num_classes,
            pth_path=path
        ).to(device)
    
    else:
        raise ValueError("Invalid MODEL_NAME")

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    device = DEVICE

    print("Device:", device)

    for dataset in DATASETS:

        print("\n===== DATASET:", dataset, "=====")

        train_loader, num_classes = get_dataset(dataset, train =True, max_samples = train_sample_number)
        test_loader, _ = get_dataset(dataset, train = False, max_samples= test_sample_number)

        model = load_model(MODEL_NAME, dataset, device)

        # Baseline
        base_accuracy, top_5, top_10, precision, recall, f1_score = evaluate(model, test_loader, device)
        print(f"{dataset.upper()} |CLEAN BASELINE|\n"
            f"Accuracy={base_accuracy:.4f}|"
            f"Top5={top_5:.4f}|"
            f"Top10={top_10:.4f}|"
            f"Precision={precision:.4f}|"
            f"Recall={recall:.4f}|"
            f"F1={f1_score:.4f}")

        # Vulnerability + Hardening 
        print("Analyzing...")
        analyzer = VulnerabilityAnalyzer(model, device)

        vuln = analyzer.analyze(train_loader, max_batches=30)
        # ===== Profiling original model =====
        min_dict, max_dict = profile_model(model, train_loader, device)

        print("Calculating Hardened Baseline")
        hardened_baseline = harden_model(model, vuln, min_dict, max_dict, ratio=HARDEN_RATIO).to(device)

        print ("Prunning...")

        if (MODEL_NAME, dataset) in PRUNING_RATIOS:
            conv_prune_ratios = PRUNING_RATIOS[(MODEL_NAME, dataset)]["conv"]
            fc_prune_ratios   = PRUNING_RATIOS[(MODEL_NAME, dataset)]["fc"]

        pruned_model , pruned_idx_dict = pruning_model(model, vuln, conv_prune_ratios, fc_prune_ratios, device)
        postprune_accuracy, pp_top_5, pp_top_10, pp_precision, pp_recall, pp_f1_score = evaluate(pruned_model, test_loader, device)
        print(f"{dataset.upper()} |Stats After Pruning|\n"
            f"Accuracy={postprune_accuracy:.4f}|"
            f"Top5={pp_top_5:.4f}|"
            f"Top10={pp_top_10:.4f}|"
            f"Precision={pp_precision:.4f}|"
            f"Recall={pp_recall:.4f}|"
            f"F1={pp_f1_score:.4f}")
        
        print("Retraining...")
        # if MODEL_NAME in ["alexnet", "vgg11", "vgg16"]:
        #     pruned_model = rebuild_first_fc(pruned_model, input_size=(3,32,32), device=device)
        pruned_model = lightweight_retraining(pruned_model, train_loader, device, epochs=10)
        retrain_accuracy, retrain_top_5, retrain_top_10, retrain_precision, retrain_recall, retrain_f1_score = evaluate(pruned_model, test_loader, device)
        print(f"{dataset.upper()} |Stats After Pruning|\n"
            f"Accuracy={retrain_accuracy:.4f}|"
            f"Top5={retrain_top_5:.4f}|"
            f"Top10={retrain_top_10:.4f}|"
            f"Precision={retrain_precision:.4f}|"
            f"Recall={retrain_recall:.4f}|"
            f"F1={retrain_f1_score:.4f}")
        
        print("Hardenning...")
        min_dict, max_dict = profile_model(pruned_model, train_loader, device)

        # Hardening
        hardened_model = harden_model_pruned(pruned_model, vuln, min_dict, max_dict, ratio=HARDEN_RATIO, pruned_idx_dict=pruned_idx_dict).to(device)
        clean_hardened_base_acc, _, _, _, _, _ = evaluate(hardened_baseline, test_loader, device)
        clean_hardened_pruned_acc, _, _, _, _, _ = evaluate(hardened_model, test_loader, device)
        # Measure performance overhead
        print("\nMeasuring Performance Overhead")
        timing_batches = get_timing_loader(train_loader, max_batches=10)

        baseline_time = measure_time(model, timing_batches, device)
        hardened_base_time = measure_time(hardened_baseline, timing_batches, device)
        hardened_pruned_time = measure_time(hardened_model, timing_batches, device)
        
        overhead_base = 100 * (hardened_base_time - baseline_time) / baseline_time
        overhead_pruned = 100 * (hardened_pruned_time - baseline_time) / baseline_time

        print(f"Baseline inference time:  {baseline_time:.2f} sec")
        print(f"Hardened baseline inference time (no pruning):  {hardened_base_time:.2f} sec")
        print(f"Hardened pruned inference time:   {hardened_pruned_time:.2f} sec")
        print(f"Overhead baseline (%):            {overhead_base:.2f}%")
        print(f"Overhead pruned (%):              {overhead_pruned:.2f}%")
        
        # CSV logging 
        csv_file = f"results/{MODEL_NAME}_{dataset}_BER_results.csv"
        
        print(f"\nLogging results to {csv_file}...")
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
            
            print("\n=== Starting Fault Injection and Evaluation ===")
            for ber in BERS:
                drops_baseline = []
                drops_pruned = []
                top5_list = []
                top10_list = []
                acc_list = []
                drop_percent_list = []
                
                print(f"\n--- BER: {ber:.0e} ---")
                for run in range(N_RUNS):
                    #Hardened baseline
                    m_base = copy.deepcopy(hardened_baseline)
                    inject_bitflips(m_base, ber)
                    accuracy, _, _, _, _, _ = evaluate(m_base, test_loader, device)
                    
                    #Hardened pruned
                    m_pruned = copy.deepcopy(hardened_model)
                    inject_bitflips(m_pruned, ber)
                    accuracy_pruned, top_5, top_10, precision, recall, f1 = evaluate(m_pruned, test_loader, device)
                    drop_base = 100 * (clean_hardened_base_acc - accuracy)
                    drop_pruned = 100 * (clean_hardened_pruned_acc - accuracy_pruned)
                    drops_baseline.append(drop_base)
                    drops_pruned.append(drop_pruned)
                    acc_list.append(accuracy_pruned)
                    top5_list.append(top_5)
                    top10_list.append(top_10)
                    drop_percent_list.append(drop_pruned)

                    writer.writerow([ber, run+1, accuracy_pruned, drop_pruned, top_5, top_10, precision, recall, f1])

                    print(f"{dataset} | BER={ber:.0e} | Run={run+1} | "
                        f"Acc={accuracy_pruned:.4f} | Top5={top_5:.4f} | "
                        f"Top10={top_10:.4f} | F1={f1:.4f} | "
                        f"Drop={drop_pruned:.2f}%")

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
                BERS,
                drop_baseline_means,
                drop_pruned_means,
                MODEL_NAME,
                dataset,
                RESULTS_DIR
            )

    plot_single_model_metric(BERS, top5_means, top5_stds, "Top5", MODEL_NAME, dataset, RESULTS_DIR)
    plot_single_model_metric(BERS, top10_means, top10_stds, "Top10", MODEL_NAME, dataset, RESULTS_DIR)
    plot_single_model_metric(BERS, acc_means, acc_stds, "Accuracy", MODEL_NAME, dataset, RESULTS_DIR)
    plot_single_model_metric(BERS, drop_means, drop_stds, "AccuracyDrop(%)", MODEL_NAME, dataset, RESULTS_DIR)
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