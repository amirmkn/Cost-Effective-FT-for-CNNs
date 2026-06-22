# Cost-Effective Fault Tolerance for CNNs Using Parameter Vulnerability Based Hardening and Pruning

This repository contains the official implementation of the paper:

> **Cost-Effective Fault Tolerance for CNNs Using Parameter Vulnerability Based Hardening and Pruning**

This framework introduces a **model-level, hardware-agnostic hardening approach** for Convolutional Neural Networks (CNNs). By analyzing the soft-error vulnerability of individual parameters, the framework selectively duplicates highly vulnerable channels and integrates an efficient **Error Detection and Correction (EDAC)** layer.

To counter the storage and computational overhead introduced by duplication, a **vulnerability-based channel pruning** technique is employed to compress the network while maintaining high fault resilience.

---

## Key Features

### Hardware-Agnostic Hardening
Enhances fault tolerance directly within the CNN architecture without requiring hardware-level modifications such as Triple Modular Redundancy (TMR).

### Vulnerability Analysis
Profiles layers and filters to identify parameters that are highly susceptible to soft errors and bit-flip faults.

### Selective Channel Duplication
Protects only the most vulnerable channels using a lightweight custom EDAC mechanism.

### Vulnerability-Based Pruning
Removes non-critical and resilient channels to reduce model size and accelerate inference by up to **24%** compared to un-pruned hardened networks.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/amirmkn/Cost-Effective-FT-for-CNNs.git
cd Cost-Effective-FT-for-CNNs
pip install -r requirements.txt
```

> **Note:** This project typically requires:
>
> - PyTorch
> - TorchVision
> - NumPy
> - A fault injection framework (e.g., `pytorch-fi`) or custom bit-flip injection modules

---

## 🛠️ Usage

### 1. Profiling & Vulnerability Analysis

Evaluate the baseline CNN to identify vulnerable filters and channels:

```bash
python profile_vulnerability.py \
    --model resnet18 \
    --dataset cifar10 \
    --error_rate 1e-4
```

---

### 2. Selective Hardening

Apply selective parameter duplication and insert the EDAC layer on identified vulnerable channels:

```bash
python harden_network.py \
    --model resnet18 \
    --vulnerability_map path/to/map.json \
    --duplication_ratio 0.15
```

---

### 3. Vulnerability-Based Pruning

Compress the hardened model to reduce memory overhead and improve inference speed:

```bash
python prune_network.py \
    --model path/to/hardened_model.pt \
    --prune_ratio 0.20
```

---

### 4. Evaluation Under Fault Injection

Simulate soft errors (bit-flips) during inference and evaluate fault resilience:

```bash
python evaluate_faults.py \
    --model path/to/hardened_pruned_model.pt \
    --error_rates 1e-5 1e-4 1e-3
```

---

## Evaluation Results

| Model Configuration | Fault Resilience (Accuracy @ High Error Rates) | Inference Speedup | Memory Overhead |
|---------------------|------------------------------------------------|-------------------|-----------------|
| Baseline CNN | Low | 1.0× | 0% |
| Triple Modular Redundancy (TMR) | High | ~0.33× | +200% |
| Hardened CNN (Ours) | High | ~0.85× | ~15% |
| Hardened + Pruned CNN (Ours) | High | ~1.10× | Negligible |

---

## 🔬 Methodology Overview

The proposed framework consists of four major stages:

1. **Vulnerability Profiling**
   - Analyze CNN parameters under soft-error injections.
   - Measure the impact of faults on model accuracy.

2. **Selective Hardening**
   - Identify highly vulnerable channels.
   - Duplicate only critical channels instead of the entire network.

3. **EDAC Integration**
   - Introduce lightweight Error Detection and Correction mechanisms.
   - Recover corrupted outputs from protected channels.

4. **Vulnerability-Based Pruning**
   - Remove low-impact channels.
   - Offset the computational and memory overhead introduced by hardening.

---

## Benefits

- Improved resilience against soft errors and bit-flip faults.
- Significantly lower overhead compared to traditional TMR approaches.
- Hardware-independent deployment.
- Reduced model size through vulnerability-aware pruning.
- Faster inference compared to hardened networks without pruning.
---

## License

This project is released under the MIT License. See the `LICENSE` file for details.

---

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request.

---

## Acknowledgements

This work builds upon research in:

- Fault-tolerant deep learning
- Soft-error resilience in neural networks
- Model compression and pruning
- CNN reliability under hardware faults
