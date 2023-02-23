import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn


def iou(pred, target, n_classes=21):
    intersections = []
    unions = []
    pred_class = torch.argmax(pred, dim=1)
    target[target == 255] = 0
    for index in range(n_classes):
        pred_index = torch.eq(pred_class, index)
        target_index = torch.eq(target, index)
        intersections.append((pred_index & target_index).sum().item())
        unions.append((pred_index | target_index).sum().item())
    return np.array(intersections), np.array(unions)


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
    def __init__(self, n_class, weight=None, smooth=1., reduction='mean'):
        super(DiceLoss, self).__init__()
        self.n_class = n_class
        self.weight = weight
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        assert input.size(1) == self.n_class

        input = torch.softmax(input, dim=1)
        input = input.permute(1, 0, 2, 3)  # C x N x H x W

        target = nn.functional.one_hot(target, num_classes=self.n_class).permute(3, 0, 1, 2)  # C x N x H x W

        input_flat = input.contiguous().view(self.n_class, -1)  # C x N*H*W
        target_flat = target.contiguous().view(self.n_class, -1)  # C x N*H*W

        numerator = (input_flat * target_flat).sum(dim=1)
        denominator = input_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice_scores = (2. * numerator + self.smooth) / (denominator + self.smooth)

        if self.weight is not None:
            assert self.weight.size(0) == self.n_class
            assert len(self.weight.shape) == 1
            class_weights = self.weight.to(device=input.device).type(input.dtype)
            class_weights.div_(class_weights.sum())
            dice_scores = class_weights * dice_scores

        loss = 1. - dice_scores

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, n_class, epsilon=1e-7):
        super(GeneralizedDiceLoss, self).__init__()
        self.n_class = n_class
        self.epsilon = epsilon

    def forward(self, input, target):
        assert input.size(1) == self.n_class
        assert len(input.shape) == 4

        input = torch.softmax(input, dim=1)
        input = input.permute(1, 0, 2, 3)  # C x N x H x W

        target = nn.functional.one_hot(target, num_classes=self.n_class).permute(3, 0, 1, 2)  # C x N x H x W

        input_flat = input.contiguous().view(self.n_class, -1)  # C x N*H*W
        target_flat = target.contiguous().view(self.n_class, -1)  # C x N*H*W

        numerator = (input_flat * target_flat).sum(dim=1)
        denominator = input_flat.pow(2).sum(dim=1) + target_flat.pow(2).sum(dim=1)

        class_weights = 1. / (torch.sum(target_flat, dim=1).pow(2) + self.epsilon)
        infs = torch.isinf(class_weights)
        class_weights[infs] = 0.
        class_weights = class_weights + infs * torch.max(class_weights)

        dice_score = (2. * torch.dot(class_weights, numerator)) / (torch.dot(class_weights, denominator))

        loss = 1. - dice_score

        return loss


class FocalDiceLoss(nn.Module):
    def __init__(self, n_class, weight=None, smooth=1., gamma=1, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.n_class = n_class
        self.weight = weight
        self.smooth = smooth
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        assert input.size(1) == self.n_class

        input = torch.softmax(input, dim=1)
        input = input.permute(1, 0, 2, 3)  # C x N x H x W

        target = nn.functional.one_hot(target, num_classes=self.n_class).permute(3, 0, 1, 2)  # C x N x H x W

        input_flat = input.contiguous().view(self.n_class, -1)  # C x N*H*W
        target_flat = target.contiguous().view(self.n_class, -1)  # C x N*H*W

        numerator = ((1 - input_flat).pow(self.gamma) * input_flat * target_flat).sum(dim=1)
        denominator = ((1 - input_flat).pow(self.gamma) * input_flat).sum(dim=1) + target_flat.sum(dim=1)

        dice_scores = (2. * numerator + self.smooth) / (denominator + self.smooth)

        if self.weight is not None:
            assert self.weight.size(0) == self.n_class
            assert len(self.weight.shape) == 1
            class_weights = self.weight.to(device=input.device).type(input.dtype)
            class_weights.div_(class_weights.sum())
            dice_scores = class_weights * dice_scores

        loss = 1. - dice_scores

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0., reduction='mean', ignore_index=-100):

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction='none', ignore_index=ignore_index)

    def forward(self, input, target):
        if input.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = input.shape[1]
            input = input.permute(0, *range(2, input.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            target = target.view(-1)
        log_p = nn.functional.log_softmax(input, dim=-1)
        cross_entropy = self.nll_loss(log_p, target)

        rows = torch.arange(len(input))
        log_pt = log_p[rows, target]

        pt = log_pt.exp()
        focal_term = (1 - pt).pow(self.gamma)

        loss = focal_term * cross_entropy

        if self.reduction == 'mean':
            loss = loss.mean()
            if self.alpha is not None:
                loss = loss / self.alpha.sum()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
