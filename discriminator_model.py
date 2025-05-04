import torch.nn as nn

class DisConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=5):
        super(DisConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, h_in = 64, w_in = 64):
        super(Discriminator, self).__init__()
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=5)
        self.relu = nn.ReLU()

        layers = []
        chan_dims = (32, 128, 256, 256)
        for i in range(1, len(chan_dims)):
            layers.append(DisConvBlock(chan_dims[i - 1], chan_dims[i], 5))
        self.cnn = nn.Sequential(*layers)
        
        fc_h = h_in - 16 # Lost 16 pixels per dimension after 4 convs of ksize 5
        fc_w = w_in - 16
        
        self.fc1 = nn.Linear(256 * fc_h * fc_w, 512)
        self.bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
