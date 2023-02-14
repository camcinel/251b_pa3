import numpy as np
import torch
import matplotlib.pyplot as plt


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


def make_plots(train_loss, train_iou, train_acc, val_loss, val_iou, val_acc, early_stop):
    fig = plt.figure(figsize=(60, 90))
    epochs = np.arange(1, len(train_loss) + 1, 1)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(epochs, train_loss, 'r', label="Training Loss")
    ax1.plot(epochs, val_loss, 'g', label="Validation Loss")
    plt.scatter(epochs[early_stop], val_loss[early_stop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=35)
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Dice Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(epochs, train_iou, 'r', label="Training IOU")
    ax2.plot(epochs, val_iou, 'g', label="Validation IOU")
    plt.scatter(epochs[early_stop], val_iou[early_stop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('IOU Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('IOU', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(epochs, train_acc, 'r', label="Training Accuracy")
    ax3.plot(epochs, val_acc, 'g', label="Validation Accuracy")
    plt.scatter(epochs[early_stop], val_acc[early_stop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=35)
    plt.yticks(fontsize=35)
    ax3.set_title('Accuracy Plots', fontsize=35.0)
    ax3.set_xlabel('Epochs', fontsize=35.0)
    ax3.set_ylabel('Accuracy', fontsize=35.0)
    ax3.legend(loc="lower right", fontsize=35.0)

    plt.show()
