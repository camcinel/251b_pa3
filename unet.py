import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        return x


class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        # Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down Blocks
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.down5 = DoubleConv(512, 1024)

        # Up Blocks
        self.up1 = DoubleConv(1024, 512)
        self.up2 = DoubleConv(512, 256)
        self.up3 = DoubleConv(256, 128)
        self.up4 = DoubleConv(128, 64)

        # Deconvolutions
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Final Layer
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.maxpool(x1))
        x3 = self.down3(self.maxpool(x2))
        x4 = self.down4(self.maxpool(x3))
        x5 = self.down5(self.maxpool(x4))

        y = torch.cat((x4, self.deconv1(x5)), dim=1)
        y = self.up1(y)
        y = torch.cat((x3, self.deconv2(y)), dim=1)
        y = self.up2(y)
        y = torch.cat((x2, self.deconv3(y)), dim=1)
        y = self.up3(y)
        y = torch.cat((x1, self.deconv4(y)), dim=1)
        y = self.up4(y)

        score = self.classifier(y)
        return score
