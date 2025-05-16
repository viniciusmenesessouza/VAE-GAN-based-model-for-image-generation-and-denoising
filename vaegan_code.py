import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from dataset_code import get_dataset_loaders
from tqdm.auto import tqdm
from utils import configure_seed, show_tensor_image
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 17})
import matplotlib.gridspec as gridspec
import numpy as np
from gan_code import Generator, Discriminator, weights_init
from main_vae import Encoder
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure

def main():
    configure_seed(seed=42)
    # from https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
    imgs_path = r'C:\Users\ruben\Documents\datasets\CelebAHQ'
    batch_size = 64
    dataloader, val_loader, img_shape = get_dataset_loaders(imgs_path, batch_size=batch_size)

    num_epochs=150
    latent_dim=100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(img_shape, latent_dim).to(device)
    decoder = Generator(nz=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    decoder.apply(weights_init)
    discriminator.apply(weights_init)

    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

    opt_E = optim.Adam(encoder.parameters(), lr=2e-4)
    opt_Dec = optim.Adam(decoder.parameters(), lr=2e-4)
    opt_Dis = optim.Adam(discriminator.parameters(), lr=2e-4)

    bce = nn.BCELoss()
    mse = nn.MSELoss(reduction='mean')

    alpha_kl = 0.1
    alpha_adv = 0.1

    step = 0
    out_vals = []
    out_metrics = []
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        mean_recon = 0.0
        mean_kl = 0.0
        mean_adv = 0.0
        count = 0
        train_loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Train", leave=True)
        for real_images in train_loop:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            step += 1

            # -------------------
            # Encode
            # -------------------
            mu, logvar = encoder(real_images)
            logvar = torch.clamp(logvar, min=-10, max=10)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            z = z.unsqueeze(-1).unsqueeze(-1)

            # -------------------
            # Decode / Generate
            # -------------------
            recon_images = decoder(z)

            # -------------------
            # Discriminator Loss
            # -------------------
            real_labels = torch.full((batch_size,), 0.9, device=device)
            fake_labels = torch.full((batch_size,), 0.1, device=device)

            real_images_noisy = real_images + 0.05 * torch.randn_like(real_images)
            recon_images_noisy = recon_images + 0.05 * torch.randn_like(recon_images)

            # if step % 1 == 0:
            for _ in range(2):
                real_output = discriminator(real_images_noisy)
                fake_output = discriminator(recon_images_noisy.detach())

                d_loss_real = bce(real_output, real_labels)
                d_loss_fake = bce(fake_output, fake_labels)
                d_loss = d_loss_real + d_loss_fake

                opt_Dis.zero_grad()
                d_loss.backward()
                opt_Dis.step()

            # -------------------
            # Generator + VAE Loss
            # -------------------
            fake_output = discriminator(recon_images_noisy)

            # Reconstruction + KL + GAN loss
            recon_loss = mse(recon_images, real_images)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            g_loss_adv = bce(fake_output, real_labels)

            total_vae_gan_loss = recon_loss + alpha_kl * min(1.0, epoch / 50) * kl_loss + alpha_adv * g_loss_adv

            train_loop.set_postfix({
                'Recon Loss': f"{recon_loss.item():.4f}",
                'KL Loss': f"{kl_loss.item():.4f}",
                'Adv Loss': f"{g_loss_adv.item():.4f}"
            })

            mean_recon += recon_loss.item()
            mean_kl += kl_loss.item()
            mean_adv += g_loss_adv.item()

            count += 1

            opt_E.zero_grad()
            opt_Dec.zero_grad()
            total_vae_gan_loss.backward()
            opt_E.step()
            opt_Dec.step()
        
        mean_recon /= count
        mean_kl /= count
        mean_adv /= count

        # Validação

        ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
        inception_score = InceptionScore().to(device)
        fid = FrechetInceptionDistance(feature=64).to(device)

        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val", leave=True)
        with torch.no_grad():
            for img in val_loop:
                noise = 0.05 * torch.randn_like(img)
                noisy = torch.clamp(img + noise, -1.0, 1.0)

                noisy = noisy.to(device, non_blocking=True)
                img = img.to(device, non_blocking=True)
                
                mu, logvar = encoder(noisy)
                logvar = torch.clamp(logvar, min=-10, max=10)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
                z = z.unsqueeze(-1).unsqueeze(-1)
                recon = decoder(z)
                
                recon_loss = nn.functional.mse_loss(recon, img, reduction='mean')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss += (recon_loss + alpha_kl*kl_loss).item()

                # Clamp to [0,1] range
                fake_imgs = (recon + 1) / 2
                real_imgs = (img + 1) / 2

                # SSIM: expects single-channel or RGB
                ssim.update(fake_imgs, real_imgs)

                real_imgs_uint8 = (real_imgs * 255).clamp(0, 255).to(torch.uint8)
                fake_imgs_uint8 = (fake_imgs * 255).clamp(0, 255).to(torch.uint8)
                # Inception Score: only fake images
                inception_score.update(fake_imgs_uint8)

                # FID: real and fake
                fid.update(real_imgs_uint8, real=True)
                fid.update(fake_imgs_uint8, real=False)

            out_metrics.append([ssim.compute().cpu(), inception_score.compute()[0].cpu(), fid.compute().cpu()])
            
            val_loss = val_loss / len(val_loader.dataset)
            
            out_vals.append([mean_recon, mean_kl, mean_adv, val_loss])

            print(f"Epoch [{epoch+1}/{num_epochs}] Recon Loss: {mean_recon:.4f} | KL Loss: {mean_kl:.4f} | Adv Loss: {mean_adv:.4f} | Val Loss: {val_loss:.4f} | SSIM: {out_metrics[-1][0]:.4f} | IS: {out_metrics[-1][1]:.4f} | FID: {out_metrics[-1][2]:.4f}")

            torch.save(decoder.state_dict(), f'vaegan_models\\vaegan_{epoch:04d}_decoder.pth')

            np.savez('vaegan_models\\loss.npz', out_vals, out_metrics)

            plt.figure(figsize=(16,4))
            side_grid = int(np.floor(img.shape[0]/3))
            gs1 = gridspec.GridSpec(3, side_grid)
            gs1.update(wspace=0, hspace=0)
            for i in range(side_grid):
                show_tensor_image(noisy[i,:], plt.subplot(gs1[i*3]))
                show_tensor_image(img[i,:], plt.subplot(gs1[i*3+1]))
                show_tensor_image(recon[i,:], plt.subplot(gs1[i*3+2]))
            plt.tight_layout()
            plt.savefig(f'vaegan_models\\vaegan_decode_epoch_{(epoch+1):04d}.jpg')
            plt.close('all')

        with torch.no_grad():
            fake = decoder(fixed_noise)

        plt.figure(figsize=(9,9))
        gs1 = gridspec.GridSpec(8, 8)
        gs1.update(wspace=0, hspace=0) 
        for i in range(8**2):
            show_tensor_image(fake[i], plt.subplot(gs1[i]))
        plt.tight_layout()
        plt.savefig(f'vaegan_models\\vaegan_fake_epoch_{(epoch+1):04d}.jpg')
        plt.close('all')

if __name__ == '__main__':
    main()
    quit()

    arr = np.load(r'vaegan_models\test2\loss.npz')['arr_0']
    print(arr[-1,1:4])
    quit()
    for i in range(arr.shape[1]):
        plt.figure()
        plt.plot(arr[:,i])
    plt.show()
    quit()

    
