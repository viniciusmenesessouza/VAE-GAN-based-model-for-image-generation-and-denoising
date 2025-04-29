import os
import argparse
from decoder_model import Decoder as Generator
from discriminator_model import Discriminator
from utils import configure_seed
from dataset_code import get_dataset_loaders

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--latent_dim', type=int, default=100, help='Size of latent dimension')
parser.add_argument('--image_size', type=int, default=64, help='Height and width of images')
parser.add_argument('--dataset_path', type=str, default=r'C:\Users\rodri\celebA\img_align_celeba\img_align_celeba')
parser.add_argument('--dataset_size', type=int, default=10, help='Number of images to use')
parser.add_argument('--noise_std', type=float, default=0.25, help='Maximum std dev of noise added')
parser.add_argument('--rect', type=bool, default=False, help='Whether to apply random rectangle occlusion')
opt = parser.parse_args()

configure_seed(seed=42)


if os.path.exists(opt.dataset_path):
    train_loader, val_loader, test_loader, img_shape = get_dataset_loaders(opt.dataset_path,
                                                                            noise_max_std=opt.noise_std,
                                                                            rect=opt.rect,
                                                                            batch_size=opt.batch_size,
                                                                            image_size=(opt.image_size, opt.image_size),
                                                                            dataset_size=opt.dataset_size)
else:
    print('Choose valid dataset path')
    quit()    

torch.set_default_device('cuda')

# Instantiate gen, dis
generator = Generator(latent_dim=opt.latent_dim)
discriminator = Discriminator()

# Set up optimizers
optimizer_generator = torch.optim.Adam(generator.parameters())
optimizer_discriminator = torch.optim.Adam(discriminator.parameters())

adversarial_loss = torch.nn.BCELoss() # Binary cross entropy

# Conduct training
for epoch in range(opt.epochs):
    for i, (images, _) in test_loader:
        '''
            Train the generator
        '''
        optimizer_generator.zero_grad()
        
        # Produce batch of images
        z = torch.rand(size=(opt.batch_size, opt.latent_dim))
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

        print(f'Epoch {epoch}/{opt.epochs}, Batch {i}/{opt.batch_size} , G_loss {loss_generator.item()}, D_loss {discriminator_loss.item()}')