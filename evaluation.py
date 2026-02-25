import torch
from sklearn.metrics import precision_recall_fscore_support

def evaluate(model, loader, device):
    """
    Evaluates the model and computes:
    Top-1, Top-5, Top-10 and Accuracy
    """
    model.eval()
    top1 = 0
    top5 = 0
    top10 = 0
    total = 0

    # Lists to store all predictions and labels for sklearn
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            # Top-K Calculations
            _, pred = out.topk(10, dim=1)
            top1 += (pred[:, 0] == y).sum().item()
            top5 += (pred[:, :5] == y.unsqueeze(1)).any(dim=1).sum().item()
            top10 += (pred == y.unsqueeze(1)).any(dim=1).sum().item()

            total += y.size(0)
            # Collect data for Precision/Recall/F1
            # We use the top-1 prediction (index 0)
            all_predictions.extend(pred[:, 0].cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Calculate Top-K Accuracies
    top1_acc = top1 / total
    top5_acc = top5 / total
    top10_acc = top10 / total

    # Calculate Precision, Recall, and F1
    # 'macro' calculates metrics for each label and finds their unweighted mean.
    # 'weighted' accounts for label imbalance.
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, 
        all_predictions, 
        average='macro', 
        zero_division=0
    )

    return {
        "accuracy": top1_acc,
        "top5": top5_acc,
        "top10": top10_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }