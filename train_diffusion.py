import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from diffusion_model import UNet  # Assumes your UNet model is in model.py
import argparse

# ----------------------------
# Dataset
# ----------------------------
class CelebAHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = list(Path(root_dir).glob("*.jpg"))
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.loader(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image

# ----------------------------
# Utilities for Diffusion
# ----------------------------
def get_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)

def q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    return sqrt_alphas_cumprod[t][:, None, None, None] * x_start + \
           sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise

def diffusion_loss(model, x0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    xt = q_sample(x0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    pred_noise = model(xt, t)
    return nn.MSELoss()(pred_noise, noise)

def sample_timestep_sinusoidal(batch_size, t_max, device, epoch, total_epochs):
    # Use a wave that shifts over epochs
    progress = epoch / total_epochs
    # phase shift over epochs
    phase = progress * 3.1416
    sampled = (torch.sin(torch.linspace(0, 3.1416, batch_size, device=device) + phase) + 1) / 2
    t = (sampled * (t_max - 1) + 1).long()
    return t

def sample_timestep_beta(batch_size, t_max, device, alpha=2.0, beta_param=2.0):
    # Sample from Beta(α, β) in [0, 1]
    beta_dist = torch.distributions.Beta(alpha, beta_param)
    sampled = beta_dist.sample((batch_size,)).to(device)
    
    # Scale to [1, t_max]
    t = (sampled * (t_max - 1)).long()
    return t

# ----------------------------
# Main training and validation
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--data_dir', type=str, default='./data/celeba_hq_256')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_1000')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Transform
    transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Dataset and split
    full_dataset = CelebAHQDataset(args.data_dir, transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model and optimizer
    model = UNet(in_ch=3, out_ch=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Noise schedule
    betas = get_beta_schedule(args.timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss_total = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        # alpha = 0.5 + (epoch / args.epochs) * 1.5  # from 0.5 to 2.0
        # beta_param = 2.0 - (epoch / args.epochs) * 1.5  # from 2.0 to 0.5

        for x0 in pbar:
            x0 = x0.to(device)
            noise = torch.randn_like(x0)
            t = torch.randint(0, args.timesteps, (x0.size(0),), device=device).long()
            # t = torch.clip(t, 1, args.timesteps - 1)
            # t = torch.randint(0, args.timesteps, (x0.size(0),), device=device).long()
            
            #t = sample_timestep_beta(x0.size(0), args.timesteps, device, alpha=alpha, beta_param=beta_param)

            loss = diffusion_loss(model, x0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * x0.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_total / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x0 in val_loader:
                x0 = x0.to(device)
                noise = torch.randn_like(x0)
                t = torch.randint(0, args.timesteps, (x0.size(0),), device=device).long()
                val_loss = diffusion_loss(model, x0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
                val_loss_total += val_loss.item() * x0.size(0)

        avg_val_loss = val_loss_total / len(val_loader.dataset)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_epoch{epoch+1}.pt"))

if __name__ == '__main__':
    main()