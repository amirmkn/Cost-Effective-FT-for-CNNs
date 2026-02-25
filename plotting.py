import matplotlib.pyplot as plt

def plot_metric(means_baseline, means_pruned, bers, metric_name, model_label):
    plt.figure(figsize=(5, 4))
    ax = plt.gca()

    # Hardened baseline (Blue Triangles)
    plt.plot(bers, means_baseline, label='Hardened baseline', 
             marker='^', color='blue', linestyle='-', markerfacecolor='blue', markersize=7)
    
    # Hardened pruned (Red Open Squares)
    plt.plot(bers, means_pruned, label='Hardened pruned', 
             marker='s', color='red', linestyle='-', markerfacecolor='none', 
             markeredgewidth=1.5, markersize=7)

    plt.xscale('log')
    plt.xlabel("BER", fontsize=12)
    plt.ylabel(f"{metric_name} (%)", fontsize=12)
    
    # Matching the specific X-axis ticks from the image
    plt.xticks(bers) 
    
    plt.grid(True, which="major", ls="--", alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Sub-caption below the X-axis (e.g., "(a) AlexNet")
    plt.xlabel(f"BER\n\n({model_label}) {MODEL_NAME.upper()}", fontsize=12)

    # Legend at the top
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), 
               ncol=2, frameon=True, edgecolor='black', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"results/{MODEL_NAME}_{metric_name}_plot.png", bbox_inches='tight')
    plt.show()
    plt.close()