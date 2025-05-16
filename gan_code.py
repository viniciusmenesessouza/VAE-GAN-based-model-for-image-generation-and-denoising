import torch.optim as optim
import torch
import torch.nn as nn
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from dataset_code import get_dataset_loaders
from tqdm.auto import tqdm
from utils import configure_seed, show_tensor_image
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 17})
import matplotlib.gridspec as gridspec
import numpy as np
import glob

class Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: Z latent vector
            nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*16),
            nn.ReLU(True),
            # 4x4
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # 8x8
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # 16x16
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # 32x32
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 64x64
            nn.ConvTranspose2d(ngf, ngf//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//2),
            nn.ReLU(True),
            # 128x128
            nn.ConvTranspose2d(ngf//2, ngf//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//4),
            nn.ReLU(True),
            # 256x256
            nn.ConvTranspose2d(ngf//4, nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: 256x256x3
            nn.Conv2d(nc, ndf//4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128x128
            nn.Conv2d(ndf//4, ndf//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf//2),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64
            nn.Conv2d(ndf//2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def evaluate_gan(dec_path, dataset_info, latent_dim):
    configure_seed(seed=42)

    _, val_loader, img_shape = dataset_info
    del dataset_info

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decoder = Generator(latent_dim).to(device)

    state_dict = torch.load(dec_path, weights_only=True)
    decoder.load_state_dict(state_dict)

    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    inception_score = InceptionScore().to(device)
    fid = FrechetInceptionDistance(feature=64).to(device)

    decoder.eval()
    with torch.no_grad():
        for real_imgs in val_loader:
            real_imgs = real_imgs.to(device, non_blocking=True)
            b_size = real_imgs.size(0)

            z = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_imgs = decoder(z)

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

            # plt.figure(figsize=(16,9))
            # for i in range(n_samples):
            #     show_tensor_image(recon[i,:], plt.subplot(3,3,i+1))
            # plt.tight_layout()
            # plt.show()
    return ssim.compute().cpu().item(), inception_score.compute()[0].cpu().item(), fid.compute().cpu().item()

def eval_gan(folder_path, batch_size):
    configure_seed(seed=42)

    # from https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
    imgs_path = r'C:\Users\ruben\Documents\datasets\CelebAHQ'
    dataset_info = get_dataset_loaders(imgs_path, batch_size=batch_size)

    path_dec = glob.glob(folder_path+'\\gan_*_decoder.pth')
    out_metrics = np.zeros((len(path_dec), 3))
    for i in range(len(path_dec)):
        metrics = evaluate_gan(path_dec[i], dataset_info, 100)
        out_metrics[i,:] = metrics
        print(i, out_metrics[i,:])
    np.savez(folder_path+'\\metrics.npz', out_metrics)

def train_gan():
    configure_seed(seed=42)
    # from https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
    imgs_path = r'C:\Users\ruben\Documents\datasets\CelebAHQ'
    batch_size = 128
    train_loader, val_loader, img_shape = get_dataset_loaders(imgs_path, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nz = 100
    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

    d_losses = []
    g_losses = []

    num_epochs = 150
    for epoch in range(num_epochs):

        g_train_loss = 0.0
        d_train_loss = 0.0
        count = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train", leave=True)
        for real_images in train_loop:
            # Train Discriminator
            netD.zero_grad()
            real_images = real_images.to(device)
            b_size = real_images.size(0)
            label = torch.full((b_size,), 1., device=device)
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            label.fill_(1.)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            g_train_loss += errG.item()
            d_train_loss += errD_real.item()+errD_fake.item()

            train_loop.set_postfix({
                    'Loss D': f"{errD_real.item()+errD_fake.item():.4f}",
                    'Loss G': f"{errG.item():.4f}"
                })
            count += 1

        d_losses.append(d_train_loss/count)
        g_losses.append(g_train_loss/count)

        print(f"Epoch {epoch+1} | "
                        f"Loss D: {d_losses[-1]:.4f} | "
                        f"Loss G: {g_losses[-1]:.4f}")
        
        torch.save(netG.state_dict(), f'gan_models\\gan_{epoch:04d}_decoder.pth')
        
        np.savez('gan_models\\gan_loss.npz', d_losses, g_losses)

        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Loss D')
        plt.plot(g_losses, label='Loss G')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        plt.savefig(f'gan_models\\a_losses.jpg')
        plt.close('all')
        
        with torch.no_grad():
            fake = netG(fixed_noise)

        plt.figure(figsize=(9,9))
        gs1 = gridspec.GridSpec(8, 8)
        gs1.update(wspace=0, hspace=0) 
        for i in range(8**2):
            show_tensor_image(fake[i], plt.subplot(gs1[i]))
        plt.tight_layout()
        plt.savefig(f'gan_models\\gan_fake_epoch_{(epoch+1):04d}.jpg')
        plt.close('all')

def train_wgan():
    configure_seed(seed=42)
    # from https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
    imgs_path = r'C:\Users\ruben\Documents\datasets\CelebAHQ'
    batch_size = 64
    train_loader, val_loader, img_shape = get_dataset_loaders(imgs_path, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nz = 100
    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Clip value for discriminator weights (WGAN)
    clip_value = 0.01
    critic_iters = 5  # Number of times to train critic per generator update

    d_losses = []
    g_losses = []

    num_epochs = 1500
    for epoch in range(num_epochs):
        g_train_loss = 0.0
        d_train_loss = 0.0
        count = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train", leave=True)
        for real_images in train_loop:
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # Train Discriminator (Critic) multiple times
            for _ in range(critic_iters):
                netD.zero_grad()

                # Real loss
                real_output = netD(real_images)
                d_loss_real = -real_output.mean()

                # Fake loss
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_images = netG(noise).detach()
                fake_output = netD(fake_images)
                d_loss_fake = fake_output.mean()

                # Total loss and backward
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizerD.step()

                # Weight clipping
                for p in netD.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # Train Generator
            netG.zero_grad()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            fake_output = netD(fake_images)
            g_loss = -fake_output.mean()

            g_loss.backward()
            optimizerG.step()

            g_train_loss += g_loss.item()
            d_train_loss += d_loss.item()

            train_loop.set_postfix({
                'Loss D': f"{d_loss.item():.4f}",
                'Loss G': f"{g_loss.item():.4f}"
            })
            count += 1

        d_losses.append(d_train_loss / count)
        g_losses.append(g_train_loss / count)

        print(f"Epoch {epoch+1} | "
            f"Loss D: {d_losses[-1]:.4f} | "
            f"Loss G: {g_losses[-1]:.4f}")
        
        torch.save(netG.state_dict(), f'gan_models\\gan_{epoch:04d}_decoder.pth')
        
        np.savez('gan_models\\gan_loss.npz', d_losses, g_losses)

        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Loss D')
        plt.plot(g_losses, label='Loss G')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        plt.savefig(f'gan_models\\a_losses.jpg')
        plt.close('all')
        
        with torch.no_grad():
            fake = netG(fixed_noise)

        plt.figure(figsize=(9,9))
        gs1 = gridspec.GridSpec(8, 8)
        gs1.update(wspace=0, hspace=0) 
        for i in range(8**2):
            show_tensor_image(fake[i], plt.subplot(gs1[i]))
        plt.tight_layout()
        plt.savefig(f'gan_models\\gan_fake_epoch_{(epoch+1):04d}.jpg')
        plt.close('all')

if __name__ == '__main__':
    # train_gan()
    # quit()
    # train_wgan()
    # quit()
    folder_path = r'gan_models\v1'
    batch_size = 128
    eval_gan(folder_path, batch_size)

    folder_path = r'gan_models\v2_wgan'
    batch_size = 128
    eval_gan(folder_path, batch_size)

    folder_path = r'gan_models\v3_wgan'
    batch_size = 64
    eval_gan(folder_path, batch_size)
    quit()

    metrics = np.load(r'gan_models\metrics.npz')['arr_0']
    plt.figure()
    plt.plot(metrics[:,0])
    plt.figure()
    plt.plot(metrics[:,1])
    plt.figure()
    plt.plot(metrics[:,2])
    plt.show()
    quit()