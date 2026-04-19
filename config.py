import torch
import os

# Device

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model / Dataset selection

MODEL_NAME = "resnet50" #[alexnet, resnet50, vgg11, vgg16]

DATASETS = [
    "cifar10","cifar100"
] #["cifar10", "cifar100","tiny_imagenet"]

train_sample_number = 2000
test_sample_number = 400

# Experiment parameters
# BERS = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4] # BERs to test
BERS = [1e-8,5e-8,1e-7,5e-7,1e-6,5e-6,1e-5,5e-5,1e-4,5e-4]

N_RUNS = 1
HARDEN_RATIO = 0.15
# Pruning
PRUNING_RATIOS = {
    # AlexNet
    ("alexnet", "cifar10"): {
        "conv": [0.05]*5,   # 5 Convolution Layers
        "fc":   [0.8]*3     # 3 FC Layers
    },
    # VGG-11
    ("vgg11", "cifar10"): {
        "conv": [0.04]*8,   # 8 Convolution Layers
        "fc":   [0.35]*3    # 3 FC Layers
    },
    # VGG-16
    ("vgg16", "cifar100"): {
        "conv": [0.01]*13,  # 13 Convolution Layers
        "fc":   [0.15]*3    # 3 FC Layers
    },
    # ResNet-50
    ("resnet50", "cifar10"): {
        "conv": [0.02]*16,  # 16 Main Convolution Layers 
        "fc":   [0.7]       # 1 Last FC Layer 
    },
    ("resnet50", "tiny_imagenet"): {
        "conv": [0.02]*16,
        "fc":   [0.8]
    },
    ("resnet50", "cifar100"): {
        "conv": [0.04]*16,  # 16 Main Convolution Layers 
        "fc":   [0.35]      # 1 Last FC Layer
    }
}
# Data settings
BATCH_SIZE = 64
NUM_WORKERS = 4
MAX_TRAIN_BATCHES = 30

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Model weight files
MODEL_WEIGHTS = {
    "alexnet": os.path.join(WEIGHTS_DIR, "alexnet_cifar10.pth"),

    "resnet50_cifar10": os.path.join(WEIGHTS_DIR, "resnet50_cifar10.pth"),
    "resnet50_cifar100": os.path.join(WEIGHTS_DIR, "resnet50_cifar100.pth"),

    "resnet50_tiny": os.path.join(WEIGHTS_DIR, "resnet50_tiny_best.pth")
}
