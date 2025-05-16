import torch
import torch.nn as nn
import math

# Sinusoidal timestep embedding
def sinusoidal_embedding(timesteps, dim):
    device = timesteps.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

# Basic conv block with optional timestep embedding
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.time_emb_proj = (
            nn.Linear(time_emb_dim, out_ch) if time_emb_dim is not None else None
        )
        self.residual_conv = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb=None):
        res = self.residual_conv(x)
        h = self.conv[0](x)
        if t_emb is not None and self.time_emb_proj is not None:
            time_emb = self.time_emb_proj(t_emb)
            time_emb = time_emb[:, :, None, None]
            h = h + time_emb
        for layer in self.conv[1:]:
            h = layer(h)
        return h + res
    


# U-Net architecture for DDPM
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Downsampling
        self.down1 = ConvBlock(in_ch, base_ch, time_emb_dim)
        self.down2 = ConvBlock(base_ch, base_ch * 2, time_emb_dim)
        self.down3 = ConvBlock(base_ch * 2, base_ch * 4, time_emb_dim)
        
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # Upsampling
        self.up3 = ConvBlock(base_ch * 8, base_ch * 2, time_emb_dim)
        self.up2 = ConvBlock(base_ch * 4, base_ch, time_emb_dim)
        self.up1 = ConvBlock(base_ch * 2, base_ch, time_emb_dim)

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x, t):
        # Embed timesteps
        t_emb = sinusoidal_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # Down
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)

        # Bottleneck
        b = self.bottleneck(self.pool(d3), t_emb)

        # Up
        u3 = self.up3(torch.cat([nn.functional.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False), d3], dim=1), t_emb)
        u2 = self.up2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False), d2], dim=1), t_emb)
        u1 = self.up1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False), d1], dim=1), t_emb)

        return self.out_conv(u1)
