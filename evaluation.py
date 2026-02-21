import torch

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

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            # گرفتن 10 تای اول
            _, pred = out.topk(10, dim=1)

            # Top-1
            top1 += (pred[:, 0] == y).sum().item()

            # Top-5 
            top5 += (pred[:, :5] == y.unsqueeze(1)).any(dim=1).sum().item()

            # Top-10
            top10 += (pred == y.unsqueeze(1)).any(dim=1).sum().item()

            total += y.size(0)

    top1_acc = top1 / total
    top5_acc = top5 / total
    top10_acc = top10 / total

    accuracy = top1_acc 

    return accuracy, top5_acc, top10_acc