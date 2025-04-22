import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(Encoder, self).__init__()
        channels = [img_size[0], 32, 64, 128, 256]
        layers = []
        for i in range(1,len(channels)):
            layers.append(ConvBlock(channels[i-1], channels[i]))
        self.cnn = nn.Sequential(*layers)

        temp_input = torch.zeros(1, img_size[0], img_size[1], img_size[2])
        temp_output = self.cnn(temp_input)
        self.flatten_size = temp_output.view(1, -1).size(1)
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
    def forward(self, x):
        x = self.cnn(x)
        
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

