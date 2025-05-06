import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, h_in=64, w_in=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # 64x64 → 32x32
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 32x32 → 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # # 16x16 → 8x8
            # nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),
            # # nn.Dropout2d(0.25),

            # # 8x8 → 4x4
            # nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2, inplace=True),
            # # nn.Dropout2d(0.25),
        )

        self.adv_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1),
            # nn.Linear(256 * 8 * 8, 1),
            # nn.Linear(512 * 4 * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        x = self.adv_head(x)
        return x
