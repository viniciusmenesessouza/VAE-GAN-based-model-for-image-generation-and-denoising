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

        self.fc = nn.Linear(latent_dim, 4 * 4 * 512)  # more channels at the start

        chan_dims = (512, 256, 128, 64, 32, 16)
        layers = []
        for i in range(1, len(chan_dims)):
            layers.append(DecConvBlock(chan_dims[i - 1], chan_dims[i]))
        self.cnn = nn.Sequential(*layers)

        self.upsample = nn.Upsample(scale_factor=2)
        self.final_pad = nn.ReflectionPad2d(1)
        self.fconv = nn.Conv2d(chan_dims[-1], 3, kernel_size=3)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)       # Start from 512×4×4
        x = self.cnn(x)                 # Now should be 16×128×128
        x = self.upsample(x)            # 16×256×256
        x = self.final_pad(x)
        x = self.fconv(x)
        x = self.output_activation(x)
        return x