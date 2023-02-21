import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml

#root = '/content/drive/MyDrive/251b_pa3/data/'
# root ='/content/drive/MyDrive/Study Material/Qtr 5/251b_pa3/data/'
root = './data/'

def plotImageMask(input, label):
    """
    Plot augmented training image and its mask in a figure.
    INPUT: Image and mask tensors
    """
    # Plot random transformaed dimage and mask
    input = np.transpose(input, (1,2,0))
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Transformed training dataset')
    ax1.imshow(input)
    ax1.set_title('Image')
    ax2.imshow(label)
    ax2.set_title('Mask')
    plt.show()

def load_config(path):
    """
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)

def iou(pred, target, n_classes=21):
    ious = []
    pred_class = torch.argmax(pred, dim=1)
    target[target == 255] = 0
    present_classes_per_mask = [torch.unique(mask) for mask in target]
    for index, class_indices in enumerate(present_classes_per_mask):
        iou_per_class = []
        for class_index in class_indices:
            pred_correct = torch.eq(pred_class[index], class_index)
            target_correct = torch.eq(target[index], class_index)
            intersection = (pred_correct & target_correct).sum().item()
            union = (pred_correct | target_correct).sum().item()
            iou_per_class.append(intersection / union)
        ious.append(np.mean(iou_per_class))
    return np.mean(ious)

    """
    Old implementation:
    
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
    """


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
    fig = plt.figure()
    epochs = np.arange(1, len(train_loss) + 1, 1)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(epochs, train_loss, 'r', label="Training Loss")
    ax1.plot(epochs, val_loss, 'g', label="Validation Loss")
    plt.scatter(epochs[early_stop], val_loss[early_stop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=11)
    plt.yticks(fontsize=11)
    ax1.set_title('Loss Plots', fontsize=11.0)
    ax1.set_xlabel('Epochs', fontsize=11.0)
    ax1.set_ylabel('Dice Loss', fontsize=11.0)
    ax1.legend(loc="upper right", fontsize=11.0)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(epochs, train_iou, 'r', label="Training IOU")
    ax2.plot(epochs, val_iou, 'g', label="Validation IOU")
    plt.scatter(epochs[early_stop], val_iou[early_stop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=11)
    plt.yticks(fontsize=11)
    ax2.set_title('IOU Plots', fontsize=11.0)
    ax2.set_xlabel('Epochs', fontsize=11.0)
    ax2.set_ylabel('IOU', fontsize=11.0)
    ax2.legend(loc="lower right", fontsize=11.0)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(epochs, train_acc, 'r', label="Training Accuracy")
    ax3.plot(epochs, val_acc, 'g', label="Validation Accuracy")
    plt.scatter(epochs[early_stop], val_acc[early_stop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=11)
    plt.yticks(fontsize=11)
    ax3.set_title('Accuracy Plots', fontsize=11.0)
    ax3.set_xlabel('Epochs', fontsize=11.0)
    ax3.set_ylabel('Accuracy', fontsize=11.0)
    ax3.legend(loc="lower right", fontsize=11.0)
    plt.savefig(root[:-5]+'my_plot.png')
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
        input = input.permute(1, 0, 2, 3) # C x N x H x W

        target = nn.functional.one_hot(target, num_classes=self.n_class).permute(3, 0, 1, 2) # C x N x H x W

        input_flat = input.contiguous().view(self.n_class, -1) # C x N*H*W
        target_flat = target.contiguous().view(self.n_class, -1) # C x N*H*W

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



class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        cross_entropy_loss = nn.functional.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-cross_entropy_loss)
        loss = (1 - pt).pow(self.gamma) * cross_entropy_loss

        if self.alpha is not None:
            alpha = self.alpha.to(device=target.get_device())
            alpha.div_(alpha.sum())
            alpha = alpha[target]
            loss = alpha * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
