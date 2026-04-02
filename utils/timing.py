import time
import torch
def get_timing_loader(loader, max_batches=10):
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i + 1 >= max_batches:
            break
    return batches

def measure_time(model, timing_batches, device, runs=5):
    model.eval()
    times = []

    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            for images, labels in timing_batches:
                images = images.to(device)
                _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)

    return sum(times) / len(times)