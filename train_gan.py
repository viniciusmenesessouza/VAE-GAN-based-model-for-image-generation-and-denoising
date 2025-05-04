import os
import argparse
from decoder_model import Decoder as Generator
from discriminator_model import Discriminator
from utils import configure_seed, show_tensor_image
from dataset_code import get_dataset_loaders
import matplotlib.pyplot as plt


import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--latent_dim', type=int, default=100, help='Size of latent dimension')
parser.add_argument('--image_size', type=int, default=64, help='Height and width of images')
parser.add_argument('--dataset_path', type=str, default=r'C:\Users\rodri\celebA\img_align_celeba\img_align_celeba')
parser.add_argument('--dataset_size', type=int, default=1000, help='Number of images to use')
parser.add_argument('--noise_std', type=float, default=0, help='Maximum std dev of noise added')
parser.add_argument('--rect', type=bool, default=False, help='Whether to apply random rectangle occlusion')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers (try 0 if out of memory error)')
parser.add_argument('--plot_fakes', type=bool, default=True, help='Whether to plot the last fake produced for each epoch')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='Adam b1 decay momentum')
parser.add_argument('--b2', type=float, default=0.999, help='Adam b2 decay momentum')
parser.add_argument('--eps', type=float, default=1e-7, help='Adam numerical instability prevention term')
opt = parser.parse_args()

configure_seed(seed=42)


if os.path.exists(opt.dataset_path):
    train_loader, _, _, img_shape = get_dataset_loaders(opt.dataset_path,
                                                                        noise_max_std=opt.noise_std,
                                                                        rect=opt.rect,
                                                                        batch_size=opt.batch_size,
                                                                        image_size=(opt.image_size, opt.image_size),
                                                                        dataset_size=opt.dataset_size,
                                                                        num_workers=opt.num_workers,
                                                                        drop_last=True)
else:
    print('Choose valid dataset path')
    quit()    

# Instantiate gen, dis
generator = Generator(latent_dim=opt.latent_dim)
discriminator = Discriminator(h_in=opt.image_size, w_in=opt.image_size)

# Set up optimizers
optimizer_generator = torch.optim.Adam(generator.parameters(),
                                        lr=opt.lr,
                                        betas=(opt.b1, opt.b2),
                                        eps=opt.eps)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),
                                           lr=opt.lr,
                                           betas=(opt.b1, opt.b2),
                                           eps=opt.eps)

adversarial_loss = torch.nn.BCELoss() # Binary cross entropy

# Conduct training
for epoch in range(opt.epochs):
    for i, (images, _) in enumerate(train_loader):
        '''
            Train the generator
        '''
        optimizer_generator.zero_grad()
        
        # Produce batch of images
        z = torch.normal(0,1,size=(opt.batch_size, opt.latent_dim))
        fakes = generator(z)

        # Goal is all ones (generator wins)
        loss_generator = adversarial_loss(discriminator(fakes), torch.ones(size=(opt.batch_size, 1)))

        # Backpropagate
        loss_generator.backward()
        optimizer_generator.step()

        '''
            Train the discriminator
        '''
        optimizer_discriminator.zero_grad()

        # Goal is all ones for reals, all zeros for fakes (discriminator wins)
        loss_reals = adversarial_loss(discriminator(images), torch.ones(size=(opt.batch_size,1)))
        loss_fakes = adversarial_loss(discriminator(fakes.detach()), torch.zeros(size=(opt.batch_size,1)))

        discriminator_loss = loss_reals + loss_fakes

        # Backpropagate
        discriminator_loss.backward()
        optimizer_discriminator.step()

        print(f'Epoch {epoch+1}/{opt.epochs}, Batch {i+1}/{len(train_loader)} , G_loss {loss_generator.item()}, D_loss {discriminator_loss.item()}')
    # Plot one of the fakes from the last batch of the epoch
    ax = plt.subplot()
    show_tensor_image(fakes[0], ax)
    plt.savefig(f'gan_fake_epoch_{epoch+1}.png')

