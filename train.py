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


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)  # xavier not applicable for biases


# TODO Get class weights
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

train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform, random_hor_flip_prob=0.5, random_vert_flip_prob=0.5)
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
    device = 'cuda'  # TODO determine which device to use (cuda or cpu)
else:
    device = 'cpu'

fcn_model = FCN(n_class=n_class).to(device=device)
fcn_model.apply(init_weights)

optimizer = torch.optim.Adam(fcn_model.parameters(), lr=learning_rate)  # TODO choose an optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
criterion = torch.nn.CrossEntropyLoss(weight=getClassWeights(train_dataset, n_classes=n_class, device=device), reduction='mean')  # TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
# criterion = torch.nn.CrossEntropyLoss()


def train(give_time=False):
    best_iou_score = 0.0
    counter = 0

    for epoch in range(epochs):
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            # TODO  reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device=device)  # TODO transfer the input to the same device as the model's
            labels = labels.to(device=device)  # TODO transfer the labels to the same device as the model's

            outputs = fcn_model.forward(inputs)  # TODO  Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

            loss = criterion(outputs, labels)  # TODO  calculate loss

            # TODO  backpropagate
            loss.backward()

            # TODO  update the weights
            optimizer.step()

            if iter % 10 == 0:
                # print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
                print('Train:\t[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, epochs, iter, len(train_loader), loss.item()))

        if give_time:
            print("Finish epoch %d\tTime elapsed %.4f seconds" % (epoch, time.time() - ts))

        current_miou_score = val(epoch)
        scheduler.step()

        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            best_model = deepcopy(fcn_model)
            best_epoch = epoch
            counter = 0
            # save the best model
        elif current_miou_score < best_iou_score:
            counter += 1
        if counter == patience:
            print(f'Early stop at epoch {epoch}\tBest epoch: {best_epoch}')
            break

    return best_model


def val(epoch):
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):
            input = input.to(device=device)
            label = label.to(device=device)

            output = fcn_model.forward(input)

            losses.append(criterion(output, label).item())
            accuracy.append(util.pixel_acc(output, label))
            mean_iou_scores.append(util.iou(output, label, n_classes=n_class))

    '''print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")'''
    print('Val:\t[%d/%d][1/1]\tLoss: %.4f\tIOU: %.4f\tAcc: %.4f'
          % (epoch, epochs, np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)))

    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores)


def modelTest():
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):
            input = input.to(device=device)
            label = label.to(device=device)

            output = fcn_model.forward(input)

            losses.append(criterion(output, label).item())
            accuracy.append(util.pixel_acc(output, label))
            mean_iou_scores.append(util.iou(output, label, n_classes=n_class))

    '''print(f"Loss is {np.mean(losses)}")
    print(f"IoU is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc is {np.mean(accuracy)}")'''
    print('Test:\t[1/1][1/1]\tLoss: %.4f\tIOU: %.4f\tAcc: %.4f'
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
    fcn_model = train(args.time)
    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
