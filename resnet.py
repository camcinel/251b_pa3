import torch
import torchvision
import torch.nn as nn


class Resnet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

        # choose resnet34 or resnet18
        resnet_mod = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)

        # remove the last two layers so it fits in our decoder framework
        self.newmodel = torch.nn.Sequential(*(list(resnet_mod.children())[:-2]))

        # freeze the model weights so they stay intact and the deconvolution weights change.
        for parameter in self.newmodel.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        # the output of the transfer learning
        x = self.newmodel(x)

        x = self.bn1(self.relu(self.deconv1(x)))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.bn5(self.relu(self.deconv5(x)))

        score = self.classifier(x)

        return score  # size=(N, n_class, H, W)
