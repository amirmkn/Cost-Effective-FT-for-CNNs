import matplotlib.pyplot as plt

# ============================================================
# Generic single-model metric plot (used for robustness curves)
# ============================================================
def plot_single_model_metric(
    bers,
    means,
    stds,
    metric_name,
    model_name,
    dataset_name,
    log_x=True,
    ylabel=None,
    save_path=None
):
    """
    Plots a single metric vs BER with optional error bars.
    """

    plt.figure(figsize=(6, 4))

    plt.errorbar(
        bers,
        means,
        yerr=stds,
        fmt='o-',
        capsize=4,
        linewidth=2
    )

    if log_x:
        plt.xscale("log")

    plt.xlabel("Bit Error Rate (BER)")
    plt.ylabel(ylabel if ylabel else metric_name)
    plt.title(f"{model_name} on {dataset_name}")

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


# ============================================================
# Paper-style comparison plot:
# Accuracy drop vs BER (baseline vs pruned)
# ============================================================
def plot_baseline_vs_pruned(
    bers,
    drop_baseline_means,
    drop_pruned_means,
    model_name,
    dataset_name,
    save_path=None
):
    """
    Replicates the IEEE-style plot:
    Accuracy Drop (%) vs BER (log-scale)
    """

    plt.figure(figsize=(6, 4))

    plt.plot(
        bers,
        drop_baseline_means,
        marker='^',
        linestyle='-',
        linewidth=2,
        label="Hardened baseline"
    )

    plt.plot(
        bers,
        drop_pruned_means,
        marker='s',
        linestyle='-',
        linewidth=2,
        label="Hardened pruned"
    )

    plt.xscale("log")
    plt.xlabel("Bit Error Rate (BER)")
    plt.ylabel("Accuracy Drop (%)")

    plt.title(f"{model_name} on {dataset_name}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


# ============================================================
# Pareto plot:
# Performance overhead vs accuracy drop
# ============================================================
def plot_pareto_overhead_vs_accuracy(
    accuracy_drops,
    overheads,
    labels,
    title,
    save_path=None
):
    """
    Pareto-style plot to show robustness/overhead tradeoff.
    """

    plt.figure(figsize=(6, 4))

    for acc_drop, overhead, label in zip(accuracy_drops, overheads, labels):
        plt.scatter(acc_drop, overhead, s=80, label=label)

    plt.xlabel("Accuracy Drop (%)")
    plt.ylabel("Performance Overhead (%)")
    plt.title(title)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
