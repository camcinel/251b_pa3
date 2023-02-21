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


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
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
                        random_vert_flip_prob=0.5)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

epochs = 100
learning_rate = 0.01
n_class = 21
patience = 25

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

fcn_model = UNet(n_class=n_class).to(device=device)
fcn_model.apply(init_weights)

optimizer = torch.optim.Adam(fcn_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
criterion = torch.nn.CrossEntropyLoss(weight=getClassWeights(train_dataset, n_classes=n_class, device=device),
                                      reduction='mean')


# criterion = torch.nn.CrossEntropyLoss()


def train(give_time=False):
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

        if give_time:
            print("Finish epoch %d\tTime elapsed %.4f seconds" % (epoch, time.time() - ts))

        current_loss, current_miou_score, current_acc = val(epoch)

        train_mean_iou = intersection / union

        train_loss.append(np.mean(epoch_loss))
        train_iou.append(np.mean(train_mean_iou))
        train_acc.append(np.mean(epoch_acc))
        val_loss.append(current_loss)
        val_iou.append(current_miou_score)
        val_acc.append(current_acc)
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

    return best_model, best_epoch, train_loss, train_iou, train_acc, val_loss, val_iou, val_acc


def val(epoch):
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    accuracy = []
    intersection = np.zeros(n_class)
    union = np.zeros(n_class)

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):
            input = input.to(device=device)
            label = label.to(device=device)

            output = fcn_model(input)

            losses.append(criterion(output, label).item())
            accuracy.append(util.pixel_acc(output, label))
            batch_intersection, batch_union = util.iou(output, label)
            intersection += batch_intersection
            union += batch_union

    mean_iou_scores = intersection / union
    print('Val:\t[%d/%d][1/1]\tLoss: %.4f\tIOU: %.4f\tAcc: %.4f'
          % (epoch, epochs, np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)))

    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)


def modelTest():
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    accuracy = []
    intersection = np.zeros(n_class)
    union = np.zeros(n_class)

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):
            input = input.to(device=device)
            label = label.to(device=device)

            output = fcn_model(input)

            losses.append(criterion(output, label).item())
            accuracy.append(util.pixel_acc(output, label))
            batch_intersection, batch_union = util.iou(output, label)
            intersection += batch_intersection
            union += batch_union

    mean_iou_scores = intersection / union
    print('Val:\t[%d/%d][1/1]\tLoss: %.4f\tIOU: %.4f\tAcc: %.4f'
          % (np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)))

    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', action='store_true',
                        help='Print time elapsed per epoch for training')
    args = parser.parse_args()

    print(f'Training on {device}')
    print('Format:\t[epoch/total epochs][mini batch/total batches]\tLoss\tIOU\tAccuracy')

    val(-1)  # show the accuracy before training
    fcn_model, best_epoch, train_loss, train_iou, train_acc, val_loss, val_iou, val_acc = train(give_time=False)
    modelTest()

    util.make_plots(train_loss, train_iou, train_acc, val_loss, val_iou, val_acc, best_epoch)
    images.make_images(fcn_model,
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
