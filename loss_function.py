import torch
import torch.nn.functional as F

def gan_loss(discriminator, real_images, generated_images):
    real_preds = discriminator(real_images)
    fake_preds = discriminator(generated_images)
    
    real_loss = F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
    fake_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
    
    return real_loss + fake_loss

def discriminator_reconstruction_loss(discriminator, real_images, reconstructed_images):
    with torch.no_grad():
        real_features = discriminator.extract_features(real_images)
    
    recon_features = discriminator.extract_features(reconstructed_images)
    
    # L2 loss between discriminator feature representations
    loss = F.mse_loss(recon_features, real_features)
    return loss

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def total_loss(discriminator, real_images, generated_images, recon_images, mu, logvar):
    gan = gan_loss(discriminator, real_images, generated_images)
    recon = discriminator_reconstruction_loss(discriminator, real_images, recon_images)
    kl = kl_divergence(mu, logvar)
    
    return recon + kl + gan
