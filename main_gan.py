import os
import argparse
from decoder_model import Decoder as Generator
from discriminator_model_v2 import Discriminator
from utils import configure_seed, show_tensor_image
from dataset_code import get_dataset_loaders
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    fake = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=200, help='Size of latent dimension')
    parser.add_argument('--image_size', type=int, default=64, help='Height and width of images')
    # parser.add_argument('--dataset_size', type=int, default=1000, help='Number of images to use')
    # parser.add_argument('--noise_std', type=float, default=0, help='Maximum std dev of noise added')
    # parser.add_argument('--rect', type=bool, default=False, help='Whether to apply random rectangle occlusion')
    parser.add_argument('--plot_fakes', type=bool, default=True, help='Whether to plot the last fake produced for each epoch')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='Adam b1 decay momentum')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam b2 decay momentum')
    parser.add_argument('--eps', type=float, default=1e-7, help='Adam numerical instability prevention term')
    opt = parser.parse_args()

    configure_seed(seed=42)

    # from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
    imgs_paths = [
        r'C:\Users\ruben\Documents\datasets\CelebA\img_align_celeba',
        r'C:\Users\rodri\celebA\img_align_celeba\img_align_celeba'
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

    if os.path.exists(imgs_path):
        train_loader, _, _, img_shape = get_dataset_loaders(imgs_path,
                                                            noise_max_std=None,
                                                            # rect=opt.rect,
                                                            batch_size=opt.batch_size,
                                                            image_size=(opt.image_size, opt.image_size),
                                                            # dataset_size=opt.dataset_size
                                                            )
    else:
        print('Choose valid dataset path')
        quit()    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate gen, dis
    generator = Generator(latent_dim=opt.latent_dim).to(device)
    discriminator = Discriminator(h_in=opt.image_size, w_in=opt.image_size).to(device)

    # Set up optimizers
    # optimizer_generator = torch.optim.Adam(generator.parameters(),
    #                                         lr=opt.lr,
    #                                         betas=(opt.b1, opt.b2),
    #                                         eps=opt.eps)
    # optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),
    #                                         lr=opt.lr,
    #                                         betas=(opt.b1, opt.b2),
    #                                         eps=opt.eps)
    
    optimizer_generator = torch.optim.Adam(generator.parameters(),lr=1e-3)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),lr=1e-3)

    adversarial_loss = nn.BCEWithLogitsLoss() # torch.nn.BCELoss() # Binary cross entropy

    out_g_loss = []
    out_d_loss = []

    # Conduct training
    for epoch in range(opt.epochs):
        generator.train()
        discriminator.train()

        g_train_loss = 0.0
        d_train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs} Train", leave=True)
        count = 0
        for images in train_loop:
            images = images.to(device, non_blocking=True)
            # images = images + torch.randn_like(images) * 0.125
            for _ in range(2):
                '''
                    Train the generator
                '''
                optimizer_generator.zero_grad()
                
                z = torch.randn(images.size(0), opt.latent_dim, device=images.device)

                fakes = generator(z)

                preds = discriminator(fakes)

                # Goal is all ones (generator wins)
                loss_generator = adversarial_loss(preds, torch.ones_like(preds)*0.9)

                # Backpropagate
                loss_generator.backward()
                optimizer_generator.step()

            '''
                Train the discriminator
            '''
            optimizer_discriminator.zero_grad()

            # Goal is all ones for reals, all zeros for fakes (discriminator wins)
            loss_reals = adversarial_loss(discriminator(images), torch.ones_like(preds)*0.9)
            loss_fakes = adversarial_loss(discriminator(fakes.detach()), torch.zeros_like(preds))

            discriminator_loss = loss_reals + loss_fakes

            # Backpropagate
            discriminator_loss.backward()
            optimizer_discriminator.step()

            g_train_loss += loss_generator.item()
            d_train_loss += discriminator_loss.item()

            train_loop.set_postfix({
                'G Loss': f"{loss_generator.item():.4f}",
                'D Loss': f"{discriminator_loss.item():.4f}"
            })
            count += 1
        
        out_g_loss.append(g_train_loss/count)
        out_d_loss.append(d_train_loss/count)
        
        print(f"Epoch {epoch+1} | "
                    f"G Loss: {out_g_loss[-1]:.4f} | "
                    f"D Loss: {out_d_loss[-1]:.4f}")
        
        np.savez('gan_models\\gan_losses.npz', out_g_loss, out_d_loss)

        # Plot one of the fakes from the last batch of the epoch
        plt.figure(figsize=(16,9))
        for i in range(9):
            show_tensor_image(fakes[i], plt.subplot(3,3,i+1))
        plt.tight_layout()
        plt.savefig(f'gan_models\\gan_fake_epoch_{(epoch+1):04d}.jpg')
        plt.close('all')

        plt.figure(figsize=(16,9))
        x = np.arange(len(out_g_loss))
        plt.plot(x, out_g_loss)
        plt.plot(x, out_d_loss)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'gan_models\\losses.jpg')
        plt.close('all')

    
        torch.save(generator.state_dict(), f'gan_models\\gan_{(epoch+1):04d}_generator.pth')

# remover sigmoide e fazer nn.BCEWithLogitsLoss()
# remover um bloco convolucional