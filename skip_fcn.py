import torch.nn as nn



class Conv(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, kernel_size=1, stride=1, padding=0, output_padding=0):

        super().__init__()
        
        self.sequence = nn.Sequential([
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, output_padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dim)
            ])

    def forward(self, x):
        return self.sequence(x)

class deConv(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, kernel_size=1, stride=1, padding=0, output_padding=0):
        
        super().__init__()
        
        self.sequence = nn.Sequential([
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dim)
            ])

    def forward(self, x):
        return self.sequence(x)

# ToDO Fill in the __ values
class skipFCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        # Convolution layers
        self.conv1 = Conv(3, 32, 3, 1, 1)           #(32,224,224)       
        
        self.conv2 = Conv(32, 64, 3, 1, 1)          #(64,224,224)       
        self.mpool1= nn.MaxPool2d(2, 2, 0)          #(64,112,112)

        self.conv3 = Conv(64, 64, 3, 1, 1)          #(64,112,112)       
        self.conv4 = Conv(64, 128, 3, 1, 1)         #(128,112,112)        
        self.mpool2 = nn.MaxPool2d(2, 2, 0)         #(128,56,56)

        self.conv5 = Conv(128, 128, 3, 1, 1)        #(128,56,56)                
        self.conv6 = Conv(128, 256, 3, 1, 1)        #(256,56,56)        
        self.mpool3 = nn.MaxPool2d(2, 2, 0)          #(256,28,28)

        self.conv7 = Conv(256, 256, 3, 1, 1)        #(256,28,28)      
        self.conv8 = Conv(256, 512, 3, 1, 1)        #(512,28,28)     
        self.mpool4 = nn.MaxPool2d(2, 2, 0)         #(512,14,14)


        # Deconvolution layers
        self.deconv1 = deConv(512, 256, 7, 2, 3, 1)     #(256,28,28)
        self.deconv2 = deConv(256, 128, 7, 2, 3, 1)     #(128,56,56)
        self.deconv3 = deConv(128, 64, 7, 2, 3, 1)      #(64,112,112)
        self.deconv4 = deConv(64, 32, 7, 2, 3, 1)       #(32,224,224)

        self.classifier = Conv(32, self.n_class, 1)     #(n_class,224,224)



    def forward(self, x):
        out1 = self.conv1(x)                                #(32,224,224)
        out2 = self.mpool1(self.conv2(out1))                #(64,112,112)
        out3 = self.mpool2(self.conv4(self.conv3(out2)))    #(128,56,56)
        out4 = self.mpool3(self.conv6(self.conv5(out3)))    #(256,28,28)
        y = self.mpool4(self.conv8(self.conv7(out4)))       #(512,14,14)

        y = self.deconv1(y) + out4                          #(256,28,28)
        y = self.deconv2(y) + out3                          #(128,56,56)
        y = self.deconv3(y) + out2                          #(64,112,112)
        y = self.deconv4(y) + out1                          #(32,224,224)


        y = self.classifier(y)        
        return y  # size=(N, n_class, H, W)
