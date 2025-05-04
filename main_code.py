# %%

import os
from utils import configure_seed, show_tensor_image
import torch
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 17})
from dataset_code import get_dataset_loaders
from encoder_model import Encoder
from decoder_model import Decoder
from torchmetrics.image.inception import InceptionScore
from torcheval.metrics import FrechetInceptionDistance
from torcheval.metrics import PeakSignalNoiseRatio
from torcheval.metrics import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.optim as optim
import torch.nn as nn
import pickle, glob

#%%

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def show_tensor_image(img_tensor, ax):
    img = img_tensor.clone().detach().cpu()
    img = img * 0.5 + 0.5  # convert back from [-1, 1] to [0, 1]
    img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
    img = img.numpy()
    ax.imshow(img)
    return ax
       
def train_vae():
    configure_seed(seed=42)

    # from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
    imgs_paths = [
        r'C:\Users\ruben\Documents\datasets\CelebA\img_align_celeba',
        r'C:\Users\rodri\celebA\img_align_celeba\img_align_celeba',
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

    train_loader, val_loader, test_loader, img_shape = get_dataset_loaders(imgs_path, noise_max_std=0.25, rect=False, batch_size=64, image_size=(64, 64))
    quit()

    # # show training images
    # for noisy, clean in train_loader:
    #     for i in range(noisy.shape[0]):
    #         show_tensor_image(noisy[i,:], plt.subplot(1,2,1))
    #         show_tensor_image(clean[i,:], plt.subplot(1,2,2))
    #         plt.show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 100
    encoder = Encoder(img_shape, latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=1e-3
    )
    vae_loss = nn.functional.mse_loss

    train_losses = []
    val_losses = []

    num_epochs = 100
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train", leave=True)
        epoch_train_loss = 0.0
        
        for noisy, clean in train_loop:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            
            # Forward pass
            mu, logvar = encoder(noisy)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            recon = decoder(z)
            
            # loss computation
            recon_loss = vae_loss(recon, clean, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + kl_loss * min(epoch/25, 1.0)
            
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
            for noisy, clean in val_loop:
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                
                mu, logvar = encoder(noisy)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
                recon = decoder(z)
                
                recon_loss = nn.functional.mse_loss(recon, clean, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss += (recon_loss + kl_loss).item()
            
            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1} | "
                    f"Train Loss: {train_losses[-1]:.4f} | "
                    f"Val Loss: {val_losses[-1]:.4f}")
            
            np.savez('vae_models\\vae_loss.npz', train_losses, val_losses)

            if len(val_losses) == 1 or np.min(val_losses) > val_loss:
                # recon = torch.clamp(recon, -1.0, 1.0)
                indx = 0 # np.random.randint(0, noisy.shape[0])
                plt.figure(figsize=(16,9))
                show_tensor_image(noisy[indx,:], plt.subplot(1,3,1))
                show_tensor_image(clean[indx,:], plt.subplot(1,3,2))
                show_tensor_image(recon[indx,:], plt.subplot(1,3,3))
                plt.tight_layout()
                plt.savefig(f'vae_models\\valimg_{epoch:04d}.jpg')
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

def get_vae_encoder_distributions(folder_path):
    configure_seed(seed=42)

    path_enc = glob.glob(folder_path+'\\vae_*_encoder.pth')[-1]

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

    train_loader, val_loader, test_loader, img_shape = get_dataset_loaders(imgs_path, noise_max_std=0.25, rect=False, batch_size=64, image_size=(64, 64))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 200
    encoder = Encoder(img_shape, latent_dim).to(device)

    state_dict = torch.load(path_enc, weights_only=True)
    encoder.load_state_dict(state_dict)

    encoder.eval()
    all_mu = []
    all_logvar = []
    with torch.no_grad():
        for noisy, clean in train_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mu, logvar = encoder(noisy)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
        for noisy, clean in val_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mu, logvar = encoder(noisy)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
        for noisy, clean in test_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mu, logvar = encoder(noisy)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
    all_mu = np.concatenate(all_mu, axis=0)
    all_logvar = np.concatenate(all_logvar, axis=0)
    np.savez('vae_encoder_distributions.npz', all_mu, all_logvar)

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

def sample_vae_decoder(folder_path, dist_npz):
    configure_seed(seed=42)

    path_dec = glob.glob(folder_path+'\\vae_*_decoder.pth')[-1]

    file = np.load(dist_npz)
    all_mu, all_logvar = file['arr_0'], file['arr_1']

    mu_bins, mu_cdfs = vals_to_hist(all_mu)
    logvar_bins, logvar_cdfs = vals_to_hist(all_logvar)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 200
    decoder = Decoder(latent_dim).to(device)
    state_dict = torch.load(path_dec, weights_only=True)
    decoder.load_state_dict(state_dict)

    n_samples = 9

    decoder.eval()
    with torch.no_grad():
        while True:
            mu = sample_distribution(mu_bins, mu_cdfs, n_samples)
            logvar = sample_distribution(logvar_bins, logvar_cdfs, n_samples)
            mu = torch.from_numpy(mu).to(device)
            logvar = torch.from_numpy(logvar).to(device)
            
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            recon = decoder(z)

            plt.figure(figsize=(16,9))
            for i in range(n_samples):
                show_tensor_image(recon[i,:], plt.subplot(3,3,i+1))
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # train_vae()
    # test_vae(r'vae_models\v3_25anneling_200dim', 200)
    # test_vae(r'vae_models\v4_noanneling_200dim', 200)
    # test_vae(r'vae_models\v5_25anneling_100dim', 100)
    # test_vae(r'vae_models\v6_25anneling_400dim', 400)
    # test_vae(r'vae_models\v7_25anneling_50dim', 50)
    # test_vae(r'vae_models\v8_25anneling_200dim_dropout', 200)
    # test_vae(r'vae_models\v9_25anneling_200dim_lessnoise', 200)
    test_vae(r'vae_models\v10_25anneling_200dim_rect', 200)
    quit()
    # get_vae_encoder_distributions(r'vae_models\v3_25anneling_200dim')
    sample_vae_decoder(r'vae_models\v3_25anneling_200dim', r'vae_encoder_distributions.npz')
    quit()


    # metrics
    inception_gt = InceptionScore().to(device)
    inception = InceptionScore().to(device)
    fid = FrechetInceptionDistance().to(device)
    psnr_gt = PeakSignalNoiseRatio().to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    
    # calculate on test set
    # for noisy, clean in test_loader:
    #     ...
    #     inception_gt.update(clean)
    #     inception.update(filtered)
    #     fid.update(clean, true) 
    #     fid.update(filtered, false) 
    #     mse.update(clean, filtered)
    #     psnr_gt.update(noisy, clean)     
    #     psnr.update(noisy, filtered)
    #     ssim.update(clean, filtered)


# %%
