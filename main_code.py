# %%

import os
from utils import configure_seed, show_tensor_image
import torch
import numpy as np
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

#%%

def main_train_VAE():
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

    train_loader, val_loader, test_loader, img_shape = get_dataset_loaders(imgs_path, noise_max_std=0.25, rect=True, batch_size=1, image_size=(64, 64), dataset_size=100)


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

    print('TRAINING')
    # training loop
    for noisy, clean in train_loader:
        print('GG')
        noisy = noisy.to(device)
        clean = clean.to(device)
        print(noisy.shape, clean.shape)
        mu, logvar = encoder(noisy)
        print(mu.shape, logvar.shape)
        std = torch.exp(0.5 * logvar)
        e_rand = torch.randn_like(mu, device=device)
        z = mu + e_rand*std
        img = decoder(z)
        print(img.shape)
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
    
if __name__ == "__main__":
    main_train_VAE()

# %%
