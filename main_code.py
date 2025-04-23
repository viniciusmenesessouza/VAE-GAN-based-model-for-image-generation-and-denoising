# %%

import os
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 17})
from dataset_code import get_dataset_loaders
from encoder_model import Encoder
from torchmetrics.image.inception import InceptionScore
from torcheval.metrics import FrechetInceptionDistance
from torcheval.metrics import PeakSignalNoiseRatio
from torcheval.metrics import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure

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

def main():
    configure_seed(seed=42)

    # from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
    imgs_path = r'C:\Users\ruben\Documents\datasets\CelebA\img_align_celeba'
    

    train_loader, val_loader, test_loader, img_shape = get_dataset_loaders(imgs_path, noise_max_std=0.25, rect=True, image_size=(64, 64))

    # # show training images
    # for noisy, clean in train_loader:
    #     for i in range(noisy.shape[0]):
    #         show_tensor_image(noisy[i,:], plt.subplot(1,2,1))
    #         show_tensor_image(clean[i,:], plt.subplot(1,2,2))
    #         plt.show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 100
    encoder = Encoder(img_shape, latent_dim).to(device)

    # training loop
    for noisy, clean in train_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        print(noisy.shape, clean.shape)
        mu, logvar = encoder(noisy)
        print(mu.shape, logvar.shape)
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
    main()

# %%
