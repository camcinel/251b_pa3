import torch
import torch.nn as nn


# ToDO Fill in the __ values
class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),         #(64,224,224)
            
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),         #(64,224,224)
            nn.MaxPool2d(2, 2, 0),      #(64,112,112)
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),        #(128,112,112)
            
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),        #(128,112,112)
            nn.MaxPool2d(2, 2, 0),      #(128,56,56)
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),        #(256,56,56)
            
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),        #(256,56,56)
            nn.MaxPool2d(2, 2, 0)       #(28, 28, 256)
          )

        # Complete the forward function for the rest of the decoder

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, self.n_class, kernel_size=1)
          )

    # TODO Complete the forward pass
    def forward(self, x):
        # Complete the forward function for the rest of the encoder
        y = self.decoder(self.encoder(x))
        
        # print(x.shape, y.shape)
        
        return y  # size=(N, n_class, H, W)
