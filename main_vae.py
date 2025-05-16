import os
from utils import configure_seed, show_tensor_image
import torch
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 17})
import matplotlib.gridspec as gridspec
from dataset_code import get_dataset_loaders
from gan_code import Generator as Decoder
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.optim as optim
import torch.nn as nn
import pickle, glob, random
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

# from https://arxiv.org/html/1610.00291v2
class Encoder(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(Encoder, self).__init__()
        channels = [img_size[0], 32, 64, 128, 256]
        layers = []
        for i in range(1,len(channels)):
            layers.append(ConvBlock(channels[i-1], channels[i]))
        self.cnn = nn.Sequential(*layers)
        
        temp_input = torch.zeros(1, img_size[0], img_size[1], img_size[2])
        temp_output = self.cnn(temp_input)
        self.flatten_size = temp_output.view(1, -1).size(1)
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
    def forward(self, x):
        x = self.cnn(x)
        
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

def train_vae():
    configure_seed(seed=42)

    # from https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
    imgs_path = r'C:\Users\ruben\Documents\datasets\CelebAHQ'
    batch_size = 64
    noise_max_std = 0.5
    train_loader, val_loader, img_shape = get_dataset_loaders(imgs_path, batch_size=batch_size)

    # # show training images
    # for img in train_loader:
    #     plt.figure(figsize=(9,9))
    #     side_grid = int(np.sqrt(img.shape[0]))
    #     gs1 = gridspec.GridSpec(side_grid, side_grid)
    #     gs1.update(wspace=0, hspace=0) 
    #     for i in range(side_grid**2):
    #         show_tensor_image(img[i], plt.subplot(gs1[i]))
    #     plt.tight_layout()
    #     plt.show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 100
    encoder = Encoder(img_shape, latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=1e-3
    )
    vae_loss = nn.MSELoss(reduction='mean')

    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

    train_losses = []
    val_losses = []

    num_epochs = 100
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train", leave=True)
        epoch_train_loss = 0.0
        
        for img in train_loop:
            noise = torch.randn_like(img) * noise_max_std
            noisy = torch.clamp(img + noise, -1.0, 1.0)

            noisy = noisy.to(device, non_blocking=True)
            img = img.to(device, non_blocking=True)
            
            # Forward pass
            mu, logvar = encoder(noisy)
            logvar = torch.clamp(logvar, min=-10, max=10)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            z = z.unsqueeze(-1).unsqueeze(-1)
            recon = decoder(z)
            
            # loss computation
            recon_loss = vae_loss(recon, img)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + kl_loss * min(epoch/50, 1.0) * 1e-5
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Atualização métricas
            epoch_train_loss += total_loss.item()
            train_loop.set_postfix({
                'Train Loss': f"{total_loss.item()/len(noisy):.4f}",
                'KL Loss': f"{kl_loss.item()/len(noisy):.4f}"
            })
        
        # Média da perda do epoch
        train_losses.append(epoch_train_loss / len(train_loader.dataset))
        
        # Validação
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val", leave=True)
        with torch.no_grad():
            for img in val_loop:
                noise = torch.randn_like(img) * torch.rand(1)*noise_max_std
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
                val_loss += (recon_loss + kl_loss).item()
            
            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1} | "
                    f"Train Loss: {train_losses[-1]:.4f} | "
                    f"Val Loss: {val_losses[-1]:.4f}")
            
            np.savez('vae_models\\vae_loss.npz', train_losses, val_losses)

            plt.figure(figsize=(16,4))
            side_grid = int(np.floor(img.shape[0]/3))
            gs1 = gridspec.GridSpec(3, side_grid)
            gs1.update(wspace=0, hspace=0)
            for i in range(side_grid):
                show_tensor_image(noisy[i,:], plt.subplot(gs1[i*3]))
                show_tensor_image(img[i,:], plt.subplot(gs1[i*3+1]))
                show_tensor_image(recon[i,:], plt.subplot(gs1[i*3+2]))
            plt.tight_layout()
            plt.savefig(f'vae_models\\valimg_{epoch:04d}.jpg')
            plt.close('all')

            with torch.no_grad():
                fake = decoder(fixed_noise)

            plt.figure(figsize=(9,9))
            gs1 = gridspec.GridSpec(8, 8)
            gs1.update(wspace=0, hspace=0) 
            for i in range(8**2):
                show_tensor_image(fake[i], plt.subplot(gs1[i]))
            plt.tight_layout()
            plt.savefig(f'vae_models\\vae_fake_epoch_{(epoch+1):04d}.jpg')
            plt.close('all')

            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid()
            plt.legend()
            plt.savefig(f'vae_models\\a_losses.jpg')
            plt.close('all')
            
            torch.save(encoder.state_dict(), f'vae_models\\vae_{epoch:04d}_encoder.pth')
            torch.save(decoder.state_dict(), f'vae_models\\vae_{epoch:04d}_decoder.pth')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def test_vae(folder_path, latent_dim):
    configure_seed(seed=42)

    path_enc = glob.glob(folder_path+'\\vae_*_encoder.pth')[-1]
    path_dec = glob.glob(folder_path+'\\vae_*_decoder.pth')[-1]

    # from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
    imgs_paths = [
        r'C:\Users\ruben\Documents\datasets\CelebA\img_align_celeba',
        r'C:\Users\rodri\celebA',
        r"dbfbfdbfdfbdbdf"
    ]
    imgs_path = None
    for path in imgs_paths:
        if os.path.exists(path):
            imgs_path = path
            break
    if imgs_path is None:
        print('Choose valid dataset path')
        quit()    

    with open(r'testloader_occlusion_obj.pkl', "rb") as f:
        test_loader = pickle.load(f)
    img_shape = [3, 64, 64]

    # train_loader, val_loader, test_loader, img_shape = get_dataset_loaders(imgs_path, noise_max_std=0.25, rect=False, batch_size=64, image_size=(64, 64))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(img_shape, latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)

    state_dict = torch.load(path_enc, weights_only=True)
    encoder.load_state_dict(state_dict)
    state_dict = torch.load(path_dec, weights_only=True)
    decoder.load_state_dict(state_dict)

    encoder.eval()
    decoder.eval()
    test_loss = 0.0
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mu, logvar = encoder(noisy)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            recon = decoder(z)

            recon_loss = nn.functional.mse_loss(recon, clean, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            test_loss += (recon_loss + kl_loss).item()
            
        test_loss = test_loss / len(test_loader.dataset)
        print('Test Loss', test_loss)

        n_imgs = 6
        for noisy, clean in test_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mu, logvar = encoder(noisy)

            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            recon = decoder(z)

            noise = torch.randn_like(recon) * 0.25
            recon_noise = recon + noise
            recon_noise = torch.clamp(recon_noise, -1.0, 1.0)
            mu2, logvar2 = encoder(recon_noise)
            z2 = mu2 + torch.exp(0.5 * logvar2) * torch.randn_like(logvar2)
            recon2 = decoder(z2)

            plt.figure(figsize=(16,9))
            for i in range(n_imgs):
                show_tensor_image(noisy[i,:], plt.subplot(3,n_imgs,i+1))
                show_tensor_image(clean[i,:], plt.subplot(3,n_imgs,i+1+n_imgs))
                show_tensor_image(recon[i,:], plt.subplot(3,n_imgs,i+1+n_imgs*2))
            plt.tight_layout()

            plt.figure(figsize=(16,9))
            for i in range(n_imgs):
                show_tensor_image(clean[i,:], plt.subplot(4,n_imgs,i+1))
                show_tensor_image(recon[i,:], plt.subplot(4,n_imgs,i+1+n_imgs))
                show_tensor_image(recon_noise[i,:], plt.subplot(4,n_imgs,i+1+n_imgs*2))
                show_tensor_image(recon2[i,:], plt.subplot(4,n_imgs,i+1+n_imgs*3))
            plt.tight_layout()

            break

        plt.show()

        test_indx = 1

        for noisy, clean in test_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mu, logvar = encoder(noisy)

            plt.figure()
            plt.plot(mu[test_indx,:].clone().detach().cpu())
            plt.figure()
            plt.plot(logvar[test_indx,:].clone().detach().cpu())

            plt.figure(figsize=(16,9))
            show_tensor_image(noisy[test_indx,:], plt.subplot(2,3,1))
            show_tensor_image(clean[test_indx,:], plt.subplot(2,3,2))
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            recon = decoder(z)
            show_tensor_image(recon[test_indx,:], plt.subplot(2,3,3))
            for i in range(3):
                mu[:,0] = mu[:,0] + 4.0
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
                recon = decoder(z)
                show_tensor_image(recon[test_indx,:], plt.subplot(2,3,i+4))
            break

        for noisy, clean in test_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mu, logvar = encoder(noisy)
            plt.figure(figsize=(16,9))
            show_tensor_image(noisy[test_indx,:], plt.subplot(2,3,1))
            show_tensor_image(clean[test_indx,:], plt.subplot(2,3,2))
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            recon = decoder(z)
            show_tensor_image(recon[test_indx,:], plt.subplot(2,3,3))
            for i in range(3):
                logvar[:,0] = logvar[:,0] + 4.0
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
                recon = decoder(z)
                show_tensor_image(recon[test_indx,:], plt.subplot(2,3,i+4))
            break

        plt.show()

def sample_vae_decoder(folder_path):
    configure_seed(seed=42)

    path_dec = glob.glob(folder_path+'\\*_decoder.pth')[-1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 100
    decoder = Decoder(latent_dim).to(device)
    state_dict = torch.load(path_dec, weights_only=True)
    decoder.load_state_dict(state_dict)

    n_samples = 3

    decoder.eval()
    with torch.no_grad():
        while True:
            z = torch.randn(n_samples**2, latent_dim).to(device)
            z = z.unsqueeze(-1).unsqueeze(-1)
            recon = decoder(z)

            plt.figure(figsize=(9,9))
            gs1 = gridspec.GridSpec(n_samples, n_samples)
            gs1.update(wspace=0, hspace=0) 
            for i in range(n_samples**2):
                show_tensor_image(recon[i], plt.subplot(gs1[i]))
            plt.tight_layout()
            plt.show()

def test_img(folder_path):
    configure_seed(seed=42)

    path_dec = glob.glob(folder_path+'\\vae_*_decoder.pth')[-1]
    path_enc = glob.glob(folder_path+'\\vae_*_encoder.pth')[-1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 100
    encoder = Encoder([3, 256, 256], latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    state_dict = torch.load(path_enc, weights_only=True)
    encoder.load_state_dict(state_dict)
    state_dict = torch.load(path_dec, weights_only=True)
    decoder.load_state_dict(state_dict)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = default_loader(r'C:\Users\ruben\Desktop\test.jpg')
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        mu, logvar = encoder(image)
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        z = z.unsqueeze(-1).unsqueeze(-1)
        recon = decoder(z)

        plt.figure(figsize=(9,9))
        show_tensor_image(image[0], plt.subplot(1,2,1))
        show_tensor_image(recon[0], plt.subplot(1,2,2))
        plt.tight_layout()
        plt.show()

def vals_to_hist(arr, n_bins=100):
    n = arr.shape[0]
    out_cdf = np.zeros((arr.shape[1], n_bins))
    out_bins = np.zeros((arr.shape[1], n_bins+1))
    for i in range(arr.shape[1]):
        freqs, bins = np.histogram(arr[:,i], bins=n_bins)
        pdf = freqs / n
        cdf = np.cumsum(pdf)
        out_cdf[i,:] = cdf
        out_bins[i,:] = bins
    return out_bins, out_cdf

def sample_distribution(bins, cdf, n_samples=1):
    out = np.zeros((n_samples,bins.shape[0]), dtype=np.float32)
    for j in range(n_samples):
        for i in range(bins.shape[0]):
            u = np.random.rand()
            idx = np.searchsorted(cdf[i,:], u)
            x0 = bins[i,idx]
            x1 = bins[i,idx + 1]
            out[j,i] = np.random.uniform(x0, x1)
    return out

def evaluate_vae(enc_path, dec_path, dataset_info, latent_dim):
    configure_seed(seed=42)

    train_loader, val_loader, img_shape = dataset_info

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(img_shape, latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)

    state_dict = torch.load(enc_path, weights_only=True)
    encoder.load_state_dict(state_dict)
    state_dict = torch.load(dec_path, weights_only=True)
    decoder.load_state_dict(state_dict)

    encoder.eval()
    all_mu = []
    all_logvar = []
    with torch.no_grad():
        for clean in train_loader:
            clean = clean.to(device, non_blocking=True)
            mu, logvar = encoder(clean)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
        for clean in val_loader:
            clean = clean.to(device, non_blocking=True)
            mu, logvar = encoder(clean)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
    all_mu = np.concatenate(all_mu, axis=0)
    all_logvar = np.concatenate(all_logvar, axis=0)

    mu_bins, mu_cdfs = vals_to_hist(all_mu)
    logvar_bins, logvar_cdfs = vals_to_hist(all_logvar)

    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    inception_score = InceptionScore().to(device)
    fid = FrechetInceptionDistance(feature=64).to(device)

    decoder.eval()
    with torch.no_grad():
        for real_imgs in val_loader:
            real_imgs = real_imgs.to(device, non_blocking=True)
            b_size = real_imgs.size(0)

            mu = sample_distribution(mu_bins, mu_cdfs, b_size)
            logvar = sample_distribution(logvar_bins, logvar_cdfs, b_size)
            mu = torch.from_numpy(mu).to(device)
            logvar = torch.from_numpy(logvar).to(device)
            
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            z = z.unsqueeze(-1).unsqueeze(-1)
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
    return ssim.compute().cpu(), inception_score.compute()[0].cpu(), fid.compute().cpu()

def eval_vae(folder_path, batch_size = 64):
    configure_seed(seed=42)

    # from https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
    imgs_path = r'C:\Users\ruben\Documents\datasets\CelebAHQ'
    
    dataset_info = get_dataset_loaders(imgs_path, batch_size=batch_size)

    # path_enc = glob.glob(folder_path+'\\*_encoder.pth')
    path_dec = glob.glob(folder_path+'\\*_decoder.pth')
    # out_metrics = np.zeros((len(path_dec), 3))
    # i = len(path_dec) - 1

    configure_seed(seed=42)

    train_loader, val_loader, img_shape = dataset_info

    latent_dim = 100
    dec_path = path_dec[-1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decoder = Decoder(latent_dim).to(device)

    state_dict = torch.load(dec_path, weights_only=True)
    decoder.load_state_dict(state_dict)

    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    inception_score = InceptionScore().to(device)
    fid = FrechetInceptionDistance(feature=64).to(device)
    inception_score_gt = InceptionScore().to(device)


    decoder.eval()
    with torch.no_grad():
        for real_imgs in val_loader:
            real_imgs = real_imgs.to(device, non_blocking=True)
            b_size = real_imgs.size(0)
            
            z = torch.randn(b_size, latent_dim).to(device)
            z = z.unsqueeze(-1).unsqueeze(-1)
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

            inception_score_gt.update(real_imgs_uint8)

            # FID: real and fake
            fid.update(real_imgs_uint8, real=True)
            fid.update(fake_imgs_uint8, real=False)

    arr = [inception_score_gt.compute()[0].cpu().item(), ssim.compute().cpu().item(), inception_score.compute()[0].cpu().item(), fid.compute().cpu().item()]
    print(arr)

def sample_vae(enc_path, dec_path, latent_dim):
    configure_seed(seed=42)

    # from https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
    imgs_path = r'C:\Users\ruben\Documents\datasets\CelebAHQ'
    batch_size = 64
    train_loader, val_loader, img_shape = get_dataset_loaders(imgs_path, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(img_shape, latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)

    state_dict = torch.load(enc_path, weights_only=True)
    encoder.load_state_dict(state_dict)
    state_dict = torch.load(dec_path, weights_only=True)
    decoder.load_state_dict(state_dict)

    encoder.eval()
    all_mu = []
    all_logvar = []
    with torch.no_grad():
        for clean in train_loader:
            clean = clean.to(device, non_blocking=True)
            mu, logvar = encoder(clean)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
        for clean in val_loader:
            clean = clean.to(device, non_blocking=True)
            mu, logvar = encoder(clean)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
    all_mu = np.concatenate(all_mu, axis=0)
    all_logvar = np.concatenate(all_logvar, axis=0)

    mu_bins, mu_cdfs = vals_to_hist(all_mu)
    logvar_bins, logvar_cdfs = vals_to_hist(all_logvar)

    decoder.eval()
    with torch.no_grad():
        while True:
            b_size = 9

            mu = sample_distribution(mu_bins, mu_cdfs, b_size)
            logvar = sample_distribution(logvar_bins, logvar_cdfs, b_size)
            mu = torch.from_numpy(mu).to(device)
            logvar = torch.from_numpy(logvar).to(device)
            
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            z = z.unsqueeze(-1).unsqueeze(-1)
            fake_imgs = decoder(z)

            plt.figure(figsize=(16,9))
            for i in range(9):
                show_tensor_image(fake_imgs[i,:], plt.subplot(3,3,i+1))
            plt.tight_layout()

            z = torch.randn_like(logvar)
            z = z.unsqueeze(-1).unsqueeze(-1)
            fake_imgs = decoder(z)

            plt.figure(figsize=(16,9))
            for i in range(9):
                show_tensor_image(fake_imgs[i,:], plt.subplot(3,3,i+1))
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # eval_vae(r'vaegan_models\test1', batch_size = 128)
    # eval_vae(r'vaegan_models\test2', batch_size = 128) # melhor
    # eval_vae(r'vaegan_models\test3', batch_size = 128)
    # eval_vae(r'vaegan_models\test4', batch_size = 128)
    # eval_vae(r'vaegan_models\test5', batch_size = 64)
    # quit()
    # sample_vae_decoder(r'vaegan_models\test2')
    # quit()
    sample_vae_decoder(r'gan_models\v1')
    quit()

    eval_vae(r'vae_models\v3')
    quit()
    sample_vae_decoder(r'vae_models\v3')
    quit()
    train_vae()
    quit()
    eval_vae()
    quit()
    # metrics = np.load(r'vae_models\v2\metrics.npz')['arr_0']
    # plt.figure()
    # plt.plot(metrics[:,0])
    # plt.figure()
    # plt.plot(metrics[:,1])
    # plt.figure()
    # plt.plot(metrics[:,2])
    # plt.show()
    # quit()

    sample_vae(r'vae_models\v2\vae_0099_encoder.pth', r'vae_models\v2\vae_0099_decoder.pth', 100)
    quit()

    sample_vae_decoder(r'vae_models\v3')
    test_img(r'vae_models\v2')
    quit()
