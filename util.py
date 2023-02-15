import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn


def iou(pred, target, n_classes=21):
    ious = []
    pred_class = torch.argmax(pred, dim=1)
    target[target == 255] = 0
    for index in range(pred.size(0)):
        ious_inst = []
        pred_class_inst = pred_class[index]
        target_inst = target[index]
        for i in range(n_classes):
            pred_class_positive = (pred_class_inst == i)  # torch.from_numpy(pred_class == i).to(device=device)
            target_positive = (target_inst == i)  # torch.from_numpy(target == i).to(device=device)
            if torch.any(target_positive):
                cap = torch.sum(pred_class_positive * target_positive)
                cup = torch.sum(pred_class_positive + target_positive)
                ious_inst.append(cap.item() / cup.item())
        ious.append(np.mean(ious_inst))
    return np.mean(ious)


def pixel_acc(pred, target):
    pred_class = torch.argmax(pred, dim=1)
    target[target == 255] = 0  # set boundary to background class
    total_correct = torch.sum(pred_class == target)
    return total_correct.item() / torch.numel(target)


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, n_class, device='cpu', epsilon=1e-7):
        super(GeneralizedDiceLoss, self).__init__()
        self.n_class = n_class
        self.device = device
        self.epsilon = epsilon

    def forward(self, input, target):
        assert input.shape[1] == self.n_class
        assert len(input.shape) == 4
        batch_size = input.size(0)
        n_class = input.size(1)
        pred = nn.functional.softmax(input, dim=1)
        target_one_hot = nn.functional.one_hot(target, num_classes=self.n_class).permute(0, 3, 1, 2).float()

        pred_flat = pred.view(batch_size, n_class, -1)
        target_one_hot_flat = target_one_hot.view(batch_size, n_class, -1)

        w = torch.sum(target_one_hot_flat, dim=(0, 2))
        w = 1 / (w * w).clamp(min=self.epsilon)

        numerator = torch.dot(w, torch.sum(pred_flat * target_one_hot_flat, dim=(0, 2)))
        denominator = torch.dot(w, torch.sum(pred_flat + target_one_hot_flat, dim=(0, 2)))
        dice_score = 2 * (numerator / denominator)
        loss = 1 - dice_score

        return loss.mean() / batch_size


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        if self.weight is None:
            self.weight = torch.ones(input.size(1))
        alpha = self.weight.type_as(input).to(device=input.get_device())
        target = target.long()

        cross_entropy_loss = nn.functional.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-cross_entropy_loss)
        alpha.div_(alpha.sum())
        at = alpha.gather(0, target.detach().view(-1)).view(target.shape)

        loss = at * (1 - pt).pow(self.gamma) * cross_entropy_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
