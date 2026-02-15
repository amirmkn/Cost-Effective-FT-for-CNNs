import torch

def evaluate(model, loader, device):
    """
    Evaluates the model on a dataset and computes Top-1
    and Top-5 accuracy.
    """
    model.eval()
    top1 = 0
    top5 = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            _, pred = out.topk(5, dim=1)

            top1 += (pred[:,0] == y).sum().item()

            for i in range(y.size(0)):
                if y[i] in pred[i]:
                    top5 += 1

            total += y.size(0)

    return top1/total, top5/total
