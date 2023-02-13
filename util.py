import numpy as np
import torch


def iou(pred, target, n_classes=21):
    ious = []
    pred_class = torch.argmax(pred, dim=1)
    target[target == 255] = 0
    for i in range(n_classes):
        pred_class_positive = (pred_class == i)  # torch.from_numpy(pred_class == i).to(device=device)
        target_positive = (target == i)  # torch.from_numpy(target == i).to(device=device)
        if torch.any(target_positive):
            cap = torch.sum(pred_class_positive * target_positive)
            cup = torch.sum(pred_class_positive + target_positive)
            ious.append(cap.item() / cup.item())
    return np.mean(ious)


def pixel_acc(pred, target):
    pred_class = torch.argmax(pred, dim=1)
    target[target == 255] = 0  # set boundary to background class
    total_correct = torch.sum(pred_class == target)
    return total_correct.item() / torch.numel(target)
