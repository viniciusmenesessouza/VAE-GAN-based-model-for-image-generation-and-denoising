import torch.nn as nn


class DecConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize = 3, stride = 1):
        super(DecConvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2) # Nearest neighbor, scale 2
        self.pad = nn.ReflectionPad2d(1) # Use a reflection pad to preserve dim
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # Nt: https://arxiv.org/html/1610.00291v2 uses latent dim 100
        
        self.fc = nn.Linear(latent_dim, 4*4*256)

        chan_dims = (256, 128, 64, 32)
        layers = []
        for i in range(1, len(chan_dims)):
            layers.append(DecConvBlock(chan_dims[i - 1], chan_dims[i]))
        self.cnn = nn.Sequential(*layers)
        # By this point image should be h32 w32 d32

        self.upsample = nn.Upsample(scale_factor = 2)
        self.final_pad = nn.ReflectionPad2d(1)
        self.fconv = nn.Conv2d(chan_dims[-1], 3, 3)
        self.output_activation = nn.Tanh()


    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)       # Outsize d256 h4 w4 
        x = self.cnn(x)                 # Out d32 h32 w32 
        x = self.upsample(x)
        x = self.final_pad(x)
        x = self.fconv(x)
        # Outputs an image d3 h64 w64 
        x = self.output_activation(x)
        return x
