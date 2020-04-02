import torch
import numpy as np

class flatten(torch.nn.Module):
#     def __init__(self, flatten_channels):
#         super(flatten, self).__init__()
#         self.flatten_channels = flatten_channels
        
    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, 1, 1, -1)

class unflatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, 1, int(np.sqrt(x.size(3))), int(np.sqrt(x.size(3))))  # torch.sqrt

class complex_conv(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        super(complex_conv, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class complex_deconv(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=2, stride=2):
        super(complex_deconv, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x


class encoder(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(encoder, self).__init__()
        
        self.complex_conv1 = complex_conv(input_channels, 16)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        
        self.complex_conv2 = complex_conv(16, 32)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        
        self.complex_conv3 = complex_conv(32, 1)
        
#         self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.flatten = flatten()
        
    def forward(self, x):
        x = self.complex_conv1(x)
        x = self.pool1(x)
        x = self.complex_conv2(x) 
        x = self.pool2(x) 
        x = self.complex_conv3(x)  
        x = self.flatten(x)
        return x

class decoder(torch.nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        
        self.unflatten = unflatten()
        self.deconv1 = complex_deconv(1, 32)
        self.deconv2 = complex_deconv(32, 16)
        self.deconv3 = complex_deconv(16, 3)
#         self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.unflatten(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

class autoencoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(autoencoder, self).__init__()

        self.encoder = encoder(input_channels)
        self.decoder = decoder()

    def forward(self, x):
        x = self.encoder()(x)
        x = self.decoder()(x)
        return x