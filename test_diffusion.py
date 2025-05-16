import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pathlib import Path
from tqdm import tqdm
from diffusion_model import UNet  
import argparse
import cv2
import numpy as np

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

def get_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)

def q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    return sqrt_alphas_cumprod[t][:, None, None, None] * x_start + \
           sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise

def main():
    # Transform
    transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_ch=3, out_ch=3).to(device)
    model.load_state_dict(torch.load("./diffusion_attn.pt", map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Noise schedule
    timesteps = 1000
    betas = get_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # Dataset and split
    full_dataset = CelebAHQDataset('./data/celeba_hq_256', transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    inception_score = InceptionScore().to(device)
    fid = FrechetInceptionDistance(feature=64).to(device)
    
    with torch.no_grad():
        for real_imgs in val_loader:
            real_imgs = real_imgs.to(device, non_blocking=True)
            b_size = real_imgs.size(0)

            z = torch.randn_like(real_imgs)
            ts = torch.randint(0, timesteps, (b_size,), device=device).long()
            xt = q_sample(real_imgs, ts, z, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            pred_noise = model(xt, ts)
            
            for i in range(ts[0].item(), -1, -1):
                t = torch.full((b_size,), i, device=device).long()
                pred_noise = model(xt, t)
                xt -= (1 - alphas[i]) / (sqrt_one_minus_alphas_cumprod[i]) * pred_noise
                xt *= 1 / torch.sqrt(alphas[i])
                xt += torch.randn_like(xt) * math.sqrt(posterior_variance[i])
            xt = torch.clamp(xt, -1, 1)
            
            fake_imgs = xt

            # Clamp to [0,1] range
            fake_imgs = (fake_imgs + 1) / 2
            real_imgs = (real_imgs + 1) / 2

            # SSIM: expects single-channel or RGB
            ssim.update(fake_imgs, real_imgs)

            real_imgs_uint8 = (real_imgs * 255).clamp(0, 255).to(torch.uint8)
            fake_imgs_uint8 = (fake_imgs * 255).clamp(0, 255).to(torch.uint8)
            # Inception Score: only fake images
            inception_score.update(fake_imgs_uint8)

            # FID: real and fake
            fid.update(real_imgs_uint8, real=True)
            fid.update(fake_imgs_uint8, real=False)
            
    # Compute metrics
    ssim_score = ssim.compute()
    inception_score_value = inception_score.compute()
    fid_score = fid.compute()
    print(f"SSIM: {ssim_score.item()}")
    print(f"Inception Score: {inception_score_value}")
    print(f"FID: {fid_score.item()}")

    with torch.no_grad():
        for x0 in val_loader:
            x0 = x0.to(device)
            # Pure noise x0
            # x0 = torch.randn_like(x0)

            # Direct sampling
            noise = torch.randn_like(x0)
            ts = torch.randint(0, timesteps, (x0.size(0),), device=device).long()
            xt = q_sample(x0, ts, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            pred_noise = model(xt, ts)
            dpred_x0 = (xt - sqrt_one_minus_alphas_cumprod[ts][:, None, None, None] * pred_noise) / sqrt_alphas_cumprod[ts][:, None, None, None]
            dpred_x0 = torch.clamp(dpred_x0, -1, 1)
            dpred_x0 = (dpred_x0 + 1) / 2
            dpred_x0 = dpred_x0.cpu().numpy()
            dpred_x0 = (dpred_x0 * 255).astype('uint8')
            dpred_x0 = cv2.cvtColor(dpred_x0[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

            # Multi-step sampling
            pred_x0 = xt.clone()
            for i in range(ts[0].item(), -1, -1):
                t = torch.full((x0.size(0),), i, device=device).long()
                pred_noise = model(pred_x0, t)
                pred_x0 -= (1 - alphas[i]) / (sqrt_one_minus_alphas_cumprod[i]) * pred_noise
                pred_x0 *= 1 / torch.sqrt(alphas[i])
                pred_x0 += torch.randn_like(pred_x0) * math.sqrt(posterior_variance[i])
            
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            pred_x0 = (pred_x0 + 1) / 2
            pred_x0 = pred_x0.cpu().numpy()
            pred_x0 = (pred_x0 * 255).astype('uint8')
            pred_x0 = cv2.cvtColor(pred_x0[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

            # Display image (original and predicted)
            x0 = (x0 + 1) / 2
            x0 = x0.cpu().numpy()
            x0 = (x0 * 255).astype('uint8')
            x0 = cv2.cvtColor(x0[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            xt = (xt + 1) / 2
            xt = xt.cpu().numpy()
            xt = (xt * 255).astype('uint8')
            xt = cv2.cvtColor(xt[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            
            combined = np.hstack((x0, xt, pred_x0, dpred_x0))
            combined = cv2.resize(combined, (combined.shape[1] // 2, combined.shape[0] // 2))
            
            cv2.imshow("Original | Noisy | Predicted - t = " + str(ts[0].item()), combined)
            cv2.waitKey(0)     
            # Save image
            # save_image(pred_x0, f"pred_x0_{t.item()}.png")
            

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()