# from basic_fcn import *
from fcn_4b import *
import time
from torch.utils.data import DataLoader
import torch
import gc
from voc import VOC
import torchvision.transforms as standard_transforms
from util import *
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


# TODO Get class weights
def getClassWeights(dataset, n_classes=21, device='cpu'):
    n_sample = torch.zeros(n_classes).to(device=device)
    total_samples = torch.zeros(1).to(device=device)
    for _, label in dataset:
        # print(torch.unique(label))
        # img_arr = input.clone().cpu().numpy()
        # img_arr = np.transpose(img_arr, (1,2,0))
        # label_arr = label.clone().cpu().numpy()
        # fig, (ax1, ax2) = plt.subplots(1,2)
        # ax1.imshow(img_arr)
        # ax2.imshow(label_arr)
        # plt.show()
        # print(np.unique(label_arr))
        # print(torch.unique(label))
        # print(torch.bincount(torch.flatten(label.to(device=device)), minlength=n_classes).shape)
        n_sample += torch.bincount(torch.flatten(label.to(device=device)), minlength=n_classes)
        total_samples += torch.numel(label.to(device=device))
    return total_samples / n_sample


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(*mean_std)
    ])
target_transform = MaskToTensor()

train_dataset = VOC('train', input_transform=input_transform, target_transform=target_transform)
val_dataset = VOC('val', input_transform=input_transform, target_transform=target_transform)
test_dataset = VOC('test', input_transform=input_transform, target_transform=target_transform)


train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)


epochs = 100
learning_rate = 0.0005
n_class = 21
patience = 25
L2 = 0.01

if torch.cuda.is_available():
    device = 'cuda'  # determine which device to use (cuda or cpu)
else:
    device = 'cpu'

fcn_model = FCN(n_class=n_class).to(device=device)
fcn_model.apply(init_weights)

optimizer = torch.optim.AdamW(fcn_model.parameters(), lr=learning_rate, weight_decay=L2)  # TODO choose an optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
criterion = torch.nn.CrossEntropyLoss(weight=getClassWeights(train_dataset, n_classes=n_class, device=device),
                                      reduction='mean')  # TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html



def train(give_time=False):
    best_iou_score = 0.0
    counter = 0
    train_loss = []
    val_loss = []
    train_iou = []
    val_iou = []
    train_acc = []
    val_acc = []
    print("start training")
    for epoch in range(epochs):
        ts = time.time()
        epoch_loss = []
        epoch_iou = []
        epoch_acc = []
        for iter, (inputs, labels) in enumerate(train_loader):
            # TODO  reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device=device)  # TODO transfer the input to the same device as the model's
            labels = labels.to(device=device)  # TODO transfer the labels to the same device as the model's
            
            # Plot random transformaed dimage and mask
            # img_arr = inputs[0].clone().cpu().numpy()
            # img_arr = np.transpose(img_arr, (1,2,0))
            # label_arr = labels[0].clone().cpu().numpy()
            # fig, (ax1, ax2) = plt.subplots(1,2)
            # ax1.imshow(img_arr)
            # ax2.imshow(label_arr)
            # plt.show()
            # break
            outputs = fcn_model(inputs)  # TODO  Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!
            # print(outputs, labels)
            loss = criterion(outputs, labels)  # TODO  calculate loss
            iou_ = iou(outputs, labels.clone().detach())
            acc = pixel_acc(outputs, labels.clone().detach())
            epoch_loss.append(loss.item())
            epoch_iou.append(iou_)
            epoch_acc.append(acc)

            # TODO  backpropagate
            loss.backward()

            # TODO  update the weights
            optimizer.step()

            # if iter % 10 == 0:
                # print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
            # print('Train:\t[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, epochs, iter, len(train_loader), loss.item()))
        
        if give_time:
            print("Finish epoch %d\tTime elapsed %.4f seconds" % (epoch, time.time() - ts))

        current_loss, current_miou_score, current_acc = val(epoch)
        train_loss.append(np.mean(epoch_loss))
        train_iou.append(np.mean(epoch_iou))
        train_acc.append(np.mean(epoch_acc))
        val_loss.append(current_loss)
        val_iou.append(current_miou_score)
        val_acc.append(current_acc)
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
        # if epoch == give_time:
        #   break
        # best_model = deepcopy(fcn_model)
        # best_epoch = epoch
    return best_model, best_epoch, train_loss, train_iou, train_acc, val_loss, val_iou, val_acc


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
            accuracy.append(pixel_acc(output, label))
            mean_iou_scores.append(iou(output, label, n_classes=n_class))

    '''print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")'''
    print('Val:\t[%d/%d][1/1]\tLoss: %.4f\tIOU: %.4f\tAcc: %.4f'
          % (epoch, epochs, np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)))

    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)


def modelTest():
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):
            input = input.to(device=device)
            label = label.to(device=device)

            output = fcn_model.forward(input)

            losses.append(criterion(output, label.long()).item())
            accuracy.append(pixel_acc(output, label))
            mean_iou_scores.append(iou(output, label, n_classes=n_class))

    '''print(f"Loss is {np.mean(losses)}")
    print(f"IoU is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc is {np.mean(accuracy)}")'''
    print('Test:\t[1/1][1/1]\tLoss: %.4f\tIOU: %.4f\tAcc: %.4f'
          % (np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)))

    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--time', action='store_true',
    #                     help='Print time elapsed per epoch for training')
    # args = parser.parse_args()

    print(f'Training on {device}')
    print('Format:\t[epoch/total epochs][mini batch/total batches]\tLoss\tIOU\tAccuracy')

    # val(-1)  # show the accuracy before training
    fcn_model, best_epoch, train_loss, train_iou, train_acc, val_loss, val_iou, val_acc = train(give_time=1)
    modelTest()
    print(learning_rate, L2)
    make_plots(train_loss, train_iou, train_acc, val_loss, val_iou, val_acc, best_epoch)
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
