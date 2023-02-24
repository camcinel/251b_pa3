from basic_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
import argparse
from copy import deepcopy
from unet import UNet
import images
from resnet import Resnet
from fc8 import FC8


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)  # xavier not applicable for biases


# Get class weights
def getClassWeights(dataset, n_classes=21, device='cpu'):
    n_sample = torch.zeros(n_classes).to(device=device)
    total_samples = torch.zeros(1).to(device=device)
    for input, label in dataset:
        n_sample += torch.bincount(torch.flatten(label.to(device=device)), minlength=n_classes)
        total_samples += torch.numel(label.to(device=device))
    return total_samples / n_sample


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])

target_transform = MaskToTensor()

train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform, random_hor_flip_prob=0.5,
                        random_vert_flip_prob=0.5, random_crop=True, rotate=True)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def train(args):
    # set up training parameters
    epochs = args.epochs
    n_class = 21
    learning_rate = args.learning_rate
    patience = args.patience

    # choose model
    if args.model == 'baseline':
        fcn_model = FCN(n_class=n_class)
    elif args.model == 'resnet':
        fcn_model = Resnet(n_class=n_class)
    elif args.model == 'unet':
        fcn_model = UNet(n_class=n_class)
    elif args.model == 'fc8':
        fcn_model = FC8(n_class=n_class)

    fcn_model.to(device=device)
    fcn_model.apply(init_weights)

    optimizer = torch.optim.Adam(fcn_model.parameters(), lr=learning_rate)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # choose loss function
    if args.weighted:
        weights = getClassWeights(train_dataset, n_classes=n_class, device=device)
    else:
        weights = None

    if args.loss == 'cross-entropy':
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    elif args.loss == 'focal':
        criterion = util.FocalLoss(alpha=weights, gamma=2)
    elif args.loss == 'dice':
        criterion = util.DiceLoss(n_class=n_class, weight=weights)


    # set-up running metrics and early stopping
    best_loss = np.inf
    counter = 0
    train_loss = []
    val_loss = []
    train_iou = []
    val_iou = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        ts = time.time()
        epoch_loss = []
        epoch_acc = []
        intersection = np.zeros(n_class)
        union = np.zeros(n_class)
        for iter, (inputs, labels) in enumerate(train_loader):
            # reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device=device)  # transfer the input to the same device as the model's
            labels = labels.to(device=device)  # transfer the labels to the same device as the model's

            outputs = fcn_model(
                inputs)  # Compute outputs

            loss = criterion(outputs, labels)  # calculate loss
            batch_intersection, batch_union = util.iou(outputs, labels.clone().detach())
            intersection += batch_intersection
            union += batch_union
            acc = util.pixel_acc(outputs, labels.clone().detach())
            epoch_loss.append(loss.item())
            epoch_acc.append(acc)

            # backpropagate
            loss.backward()

            # update the weights
            optimizer.step()

            if iter % 10 == 0:
                # print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
                print('Train:\t[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, epochs, iter, len(train_loader), loss.item()))

        if args.time:
            print("Finish epoch %d\tTime elapsed %.4f seconds" % (epoch, time.time() - ts))

        current_loss, current_miou_score, current_acc = val(epoch, epochs, fcn_model, criterion)

        train_mean_iou = intersection / union

        train_loss.append(np.mean(epoch_loss))
        train_iou.append(np.mean(train_mean_iou))
        train_acc.append(np.mean(epoch_acc))
        val_loss.append(current_loss)
        val_iou.append(current_miou_score)
        val_acc.append(current_acc)
        if args.scheduler:
            scheduler.step()

        if current_loss < best_loss:
            best_loss = current_loss
            best_model = deepcopy(fcn_model)
            best_epoch = epoch
            counter = 0
            # save the best model
        elif current_loss > best_loss:
            counter += 1
        if counter == patience:
            print(f'Early stop at epoch {epoch}\tBest epoch: {best_epoch}')
            break

    return best_model, criterion, best_epoch, train_loss, train_iou, train_acc, val_loss, val_iou, val_acc


def val(epoch, epochs, model, criterion, n_class=21):
    model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    accuracy = []
    intersection = np.zeros(n_class)
    union = np.zeros(n_class)

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):
            input = input.to(device=device)
            label = label.to(device=device)

            output = model(input)

            losses.append(criterion(output, label).item())
            accuracy.append(util.pixel_acc(output, label))
            batch_intersection, batch_union = util.iou(output, label)
            intersection += batch_intersection
            union += batch_union

    mean_iou_scores = intersection / union
    print('Val:\t[%d/%d][1/1]\tLoss: %.4f\tIOU: %.4f\tAcc: %.4f'
          % (epoch, epochs, np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)))

    model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)


def modelTest(model, criterion, n_class=21):
    model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    accuracy = []
    intersection = np.zeros(n_class)
    union = np.zeros(n_class)

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):
            input = input.to(device=device)
            label = label.to(device=device)

            output = model(input)

            losses.append(criterion(output, label).item())
            accuracy.append(util.pixel_acc(output, label))
            batch_intersection, batch_union = util.iou(output, label)
            intersection += batch_intersection
            union += batch_union

    mean_iou_scores = intersection / union
    print('Test:\t[1/1][1/1]\tLoss: %.4f\tIOU: %.4f\tAcc: %.4f'
          % (np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)))

    model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', action='store_true',
                        help='Print time elapsed per epoch for training')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='maximum number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.005,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--model', choices=['baseline', 'resnet', 'unet', 'fc8'],
                        default='baseline',
                        help='choose model to train and test')
    parser.add_argument('--weighted', action='store_true',
                        help='weight the loss function')
    parser.add_argument('--loss', choices=['cross-entropy', 'focal', 'dice'],
                        default='cross-entropy',
                        help='choose loss function')
    parser.add_argument('--patience', type=int, default=10,
                        help='choose patience for early stopping')
    parser.add_argument('--scheduler', action='store_true',
                        help='use cosine annealing scheduler')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    print(f'Training on {device}')
    print('Format:\t[epoch/total epochs][mini batch/total batches]\tLoss\tIOU\tAccuracy')

    best_model, loss_fn, best_epoch, train_loss, train_iou, train_acc, val_loss, val_iou, val_acc = train(args)
    modelTest(best_model, loss_fn)

    util.make_plots(train_loss, train_iou, train_acc, val_loss, val_iou, val_acc, best_epoch)
    images.make_images(best_model,
                       val_dataset,
                       palette=[0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
                                128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
                                64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64,
                                128],
                       index=1,
                       device=device
                       )

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
