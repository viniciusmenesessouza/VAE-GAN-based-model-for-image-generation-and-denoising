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
    
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)

        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(C), dim=2)
        h = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return self.proj(h) + x
    
class CrossAttentionToGlobalTokens(nn.Module):
    def __init__(self, dim, num_tokens=64, token_dim=256):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.to_q = nn.Linear(dim, token_dim)
        self.to_kv = nn.Linear(token_dim, token_dim * 2)
        self.proj = nn.Linear(token_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.global_tokens = nn.Parameter(torch.randn(1, num_tokens, token_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        q = self.to_q(self.norm(x_))  # [B, H*W, token_dim]
        
        tokens = self.global_tokens.expand(B, -1, -1)  # [B, num_tokens, token_dim]
        k, v = self.to_kv(tokens).chunk(2, dim=-1)  # both [B, num_tokens, token_dim]
        
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.token_dim), dim=-1)  # [B, H*W, num_tokens]
        h = torch.bmm(attn, v)  # [B, H*W, token_dim]
        h = self.proj(h).permute(0, 2, 1).view(B, C, H, W)
        return h + x  # residual connection

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

        self.cross_attn2 = CrossAttentionToGlobalTokens(base_ch * 2, num_tokens=64)
        self.cross_attn3 = CrossAttentionToGlobalTokens(base_ch * 4, num_tokens=64)
        self.attn3 = AttentionBlock(base_ch * 4)  # At 16x16
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.attn_bottleneck = AttentionBlock(base_ch * 4)  # Optional at 8x8

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
        d2 = self.cross_attn2(d2)
        d3 = self.down3(self.pool(d2), t_emb)
        d3 = self.cross_attn3(d3)

        # Bottleneck
        b = self.bottleneck(self.pool(d3), t_emb)
        b = self.attn_bottleneck(b)

        # Up
        u3 = self.up3(torch.cat([nn.functional.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False), d3], dim=1), t_emb)
        u2 = self.up2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False), d2], dim=1), t_emb)
        u1 = self.up1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False), d1], dim=1), t_emb)

        return self.out_conv(u1)
