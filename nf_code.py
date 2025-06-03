import torch
import torch.nn as nn
import math
from tqdm import tqdm
from dataset_code import get_dataset_loaders
from torchvision.utils import save_image

import torch
from torch import nn
from math import log, pi

'''
    GLOW MODEL IMPLEMENTATION
    Originally by:
    https://github.com/rosinality/glow-pytorch (Code available under MIT License)
    - Updated to use only torch linalg instead of scipy/numpy
    - Implemented an evaluation loop with Inception Score, SSIM, FID
'''

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class ActivationNormalization(nn.Module):
    '''
        Implements Activation Normalization layer
        Tested -> working as intended

        Arguments:
            in_channel -> number of channels of input
            log_det -> Boolean, whether to return log of determinant
        
        Methods:
            forward -> Forward pass given input
            reverse -> Reverse pass given output
            __initialize (private) -> Initializes w/ shapes given input 
    '''
    def __init__(self, in_channel, log_det = True):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.s = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.log_det = log_det
        self.initialized = False

    def __initialize(self, input):
        '''
        Initialize ActivationNormalization parameters with adequate shapes 
        '''
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.b.data.copy_(-mean)
            self.s.data.copy_(1 / (std + 1e-6))

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]

        if not self.initialized:
            self.__initialize(x)
            self.initialized = True

        # Output =  Scale * (Input + Shift)

        x = self.s * (x + self.b)

        if self.log_det:
            return x,  h * w * torch.sum(torch.log(torch.abs(self.s)))
        else:
            return x
    
    def reverse(self, y):
        # Input = (Output / Scale) - Shift
        return (y / self.s) - self.b


class ZeroConv2d(nn.Module):
    '''
        Implements zero-initalized convolution (foward pass)
        Tested -> Working as intended

        Arguments:
            in_channel -> number of channels of input
            out_channel -> number of channels of output
        
        Methods:
            forward -> Forward pass given input
    '''
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = nn.functional.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class InvertibleConv(nn.Module):
    '''
        Implements Inverted 2D Convolution
        Tested -> Working as intended

        Arguments:
            in_channel -> number of channels of input
            log_det -> Boolean, whether to return log determinant
        
        Methods:
            forward -> Forward pass given input
            reverse -> Reverse pass given output
    '''
    def __init__(self, in_channel, log_det = True):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        w = torch.randn(in_channel, in_channel, 
                        device = device)
        # Perform LU Decomposition so determinant computation is faster
        q, _ = torch.linalg.qr(w)
        P, L, U = torch.linalg.lu(q)
        S = torch.diag(U)
        U = torch.triu(U, 1)

        self.P = P # P is not optimized
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        self.S = nn.Parameter(torch.log(torch.abs(S)))

        # Upper, lower triangular masks
        self.U_mask = torch.triu(torch.ones_like(U, device=device),1)
        self.L_mask = self.U_mask.T
        self.S_sign = torch.sign(S)
        self.L_eye = torch.eye(self.L_mask.shape[0], device=device)

        self.log_det = log_det

    def get_weight(self):
        # Weight is P @ L @ (U + diag(s))
        w = (self.P @
             (self.L * self.L_mask + self.L_eye) @ 
             ((self.U * self.U_mask) + torch.diag(self.S_sign * torch.exp(self.S))))
        # Additional dimensions needed for conv
        return w.unsqueeze(2).unsqueeze(3)
         
    
    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        
        weight = self.get_weight()
        
        x = nn.functional.conv2d(x, weight)
        
        log_det = h * w * torch.sum(self.S)
       
        if self.log_det:
            return x, log_det
        else:
            return x
    
    def reverse(self, y):
        weight = self.get_weight()
        return nn.functional.conv2d(y, weight.squeeze().pinverse().unsqueeze(2).unsqueeze(3))


class AffineCoupling(nn.Module):
    '''
        Implements Affine Coupling layer
        Tested -> working only with additive coupling

        Arguments:
            in_channel -> Number of channels of input
            filter_size -> Number of conv filters

        Methods:
            forward -> Forward pass given input
            reverse -> Reverse pass given output
    '''
    def __init__(self, in_channel, filter_size=512, ksize=3):
        super().__init__()

        self.neuralnet = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            # Glow authors initialize last conv w/ zeros
            ZeroConv2d(filter_size, in_channel // 2)
        )

        self.neuralnet[0].weight.data.normal_(0, 0.05)
        self.neuralnet[0].bias.data.zero_()

        self.neuralnet[2].weight.data.normal_(0, 0.05)
        self.neuralnet[2].bias.data.zero_()


    def forward(self, x):
        # x is first split
        x_a, x_b = x.chunk(2, 1)

        net_out = self.neuralnet(x_a)
        # In additive coupling, output is simply summed
        out_b = x_b + net_out
        # Log Det in additive coupling is 0
        logdet = 0

        return torch.cat([x_a, out_b], 1), logdet
         
    def reverse(self, y):
        # y is split
        y_a, y_b = y.chunk(2,1)
        # NN is applied to one split
        net_out = self.neuralnet(y_a)
        # In additive coupling, output is simply subtracted
        in_b = y_b - net_out

        return torch.cat([y_a, in_b], 1)

'''
    Less verbose implementations I made, to be tested once first results are in
    Feature slight changes to implementation to be closer to original proposal
'''
class myActNorm(nn.Module):
    def __init__(self, in_channel):
        '''
        "performs an affine transformation of the activationsusing a scale and bias parameter per channel"
        '''
        self.s = nn.Parameter(torch.zeros(1,in_channel,1,1))
        self.b = nn.Parameter(torch.zeros(1,in_channel,1,1))
    
    def initialize(self, x):
        '''
        "These parameters areinitialized such that the post-actnorm activations per-channel have zero mean and unit variance"
        '''
        with torch.no_grad():
            flat_x = x.permute(1,0,2,3).contiguous().view(x.shape[1], -1)
            mean = (flat_x.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1,0,2,3))
            std = (flat_x.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1,0,2,3))

            # x * s + b should result in x with mean 0 and std 1, so
            self.s.data.copy_(1 / (std + 1e-5))
            self.b.data.copy_(-mean)

    def forward(self, x):
        # Returns y = x * s + b, log-determinant h * w * sum(log(abs(s)))
        return self.s * x + self.b, x.shape[-2] * x.shape[-1] * torch.sum(torch.log(torch.abs(self.s)))

    def reverse(self, y):
        # Returns x = (y - b)/s
        return (y - self.b)/ self.s
    

class myInvConv(nn.Module):
    def __init__(self, in_channel):
        #  first sampling a random rotation matrix W (c by c)
        # QR decomp ensures this is the case
        W, _ = torch.linalg.qr(torch.randn(in_channel, in_channel, device='cuda'))
        # PLU decomposition
        P, L, U = torch.linalg.lu(W)

        self.P = P # P is NOT trainable (permutation matrix)
        self.L = nn.Parameter(L)

        # Separate U into the upper triangular and its diagonal (s)
        s = torch.diag(U)
        U = torch.triu(U, 1)

        self.s = nn.Parameter(s)
        self.U = nn.Parameter(U)

    def compute_W(self):
        # W is given by 
        return self.P @ self.L @ (self.U + self.s)

    def forward(self, x):
        # Return y = conv2d(x, W), sum(log(abs(s)))
        return nn.functional.conv2d(x, self.compute_W()), torch.sum(torch.log(torch.abs(self.s)))

    def reverse(self, y):
        # Return x = conv2d(x, W^-1)
        return nn.functional.conv_transpose2d(y, self.compute_W())


class myAffine(nn.Module):
    def __init__(self, in_channel, filters = 512):

        '''
         "We let each NN()have three convolutional layers, 
         where the two hidden layers have ReLU activation
         functions and 512 channels. The first and last 
         convolutions are 3 by 3, while the center convolution 
         is 1  by 1"
        '''
        self.NN = nn.Sequential(
            nn.Conv2d(in_channel, filters, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, in_channel, 3)
        )

        self.NN[0].weight.data.copy_(0)
        self.NN[0].bias.data.copy_(0)


class Flow(nn.Module):
    '''
        Implements a flow of the network
        The flow consists of steps actnorm -> invconv -> affinecoupling
        Tested -> Working as intended
        
        Arguments:
            in_channel -> number of channels of input
            affine -> Boolean, whether transformation is affine
        
        Methods:
            forward -> Forward pass given input
            reverse -> Reverse pass given output
    '''
    def __init__(self, in_channel):
        super().__init__()
        self.ActNorm = ActivationNormalization(in_channel)
        self.InvConv = InvertibleConv(in_channel)
        self.AffCoupling = AffineCoupling(in_channel)
    
    def forward(self, x):

        y, log_det = self.ActNorm(x)
        y, inv_det = self.InvConv(y)
        y, aff_det = self.AffCoupling(y)
        
        log_det = log_det + inv_det
        if aff_det is not None:
            log_det = log_det + aff_det
        return y, log_det

    def reverse(self, y):
        y = self.AffCoupling.reverse(y)
        y = self.InvConv.reverse(y)
        x = self.ActNorm.reverse(y)
        return x


class FlowBlock(nn.Module):
    '''
        Implements Block with K flows
        Tested -> Working as intended

        Arguments:
            in_channel -> number of channels of input
            K -> number of flows in block
        
        Methods:
            forward -> Forward pass given input
            reverse -> Reverse pass given output
        
    '''
    def __init__(self, in_channel, K, split = True):
        super().__init__()

        # Stack K flows
        self.flows = nn.ModuleList()
        for _ in range(K):
            self.flows.append(Flow(in_channel * 4))
        
        # Whether to split at the end of the block
        self.split = split
        
        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)
        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)
        

    def forward(self, x):
        batch_size, n_channels, h, w = x.shape
        squeeze = x.view(batch_size, n_channels, h // 2, 2, w // 2, 2)
        squeeze = squeeze.permute(0, 1, 3, 5, 2, 4)
        
        y = squeeze.contiguous().view(batch_size, n_channels * 4, h // 2, w // 2)

        log_det = 0
        for flow in self.flows:
            y, ld = flow(y)
            log_det += ld

        if self.split:
            y, z_new = y.chunk(2, 1)
            mean, std = self.prior(y).chunk(2,1)
            log_prob = gaussian_log_p(z_new, mean, std)
            log_prob = log_prob.view(batch_size, -1).sum(1)
        else:
            zero = torch.zeros_like(y)
            mean, std = self.prior(zero).chunk(2,1)
            log_prob = gaussian_log_p(y, mean, std)
            log_prob = log_prob.view(batch_size, -1).sum(1)
            z_new = y

        return y, log_det, log_prob, z_new

    def reverse(self, y, eps=None, reconstruct=False):
        x = y

        if reconstruct:
            if self.split:
                x = torch.cat([y, eps], 1)
            else:
                x = eps
        else:
            if self.split:
                mean, std = self.prior(x).chunk(2,1)
                z = gaussian_sample(eps, mean, std)
                x = torch.cat([y, z], 1)
            else:
                zero = torch.zeros_like(x)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                x = z
        # x still exists
        for flow in self.flows[::-1]:
            x = flow.reverse(x)
            breakpoint()

        batch_size, num_channels, h, w = x.shape
        unsqueeze = x.view(batch_size, num_channels // 4, 2, 2, h, w)
        unsqueeze = unsqueeze.permute(0, 1, 4, 2, 5, 3)
        unsqueeze = unsqueeze.contiguous().view(
            batch_size, num_channels // 4, h * 2, w * 2
        )

        return unsqueeze


class GlowModel(nn.Module):
    '''
        Implements the full Glow model as proposed in arXiv:1807.03039
        It is comprised of L blocks with K flows each
        Tested -> Working as intended

        Arguments:
            in_channel -> number of channels of input
            K -> number of flows per block
            L -> number of blocks
    '''
    def __init__(self, in_channel, K, L):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        self.n_blocks = L
        self.n_flows = K

        num_channels = in_channel
        for _ in range(L - 1):
            self.blocks.append(FlowBlock(num_channels, K))
            num_channels *= 2
        # Note that the final block does not feature a split
        self.blocks.append(FlowBlock(num_channels, K, split=False))

    def forward(self, x):
        log_prob_sum = 0
        log_det = 0
        z_outs = []

        for block in self.blocks:
            x, l_d, l_p, z_new = block(x)
            z_outs.append(z_new)
            log_det += l_d
            if l_p is not None:
                log_prob_sum += l_p            
           
        return log_prob_sum, log_det, z_outs
    
    def reverse(self, y, reconstruct=False):
        x = None
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.reverse(y[-1], y[-1], reconstruct=reconstruct)
            else:
                x = block.reverse(x, y[-( i + 1)], reconstruct=reconstruct)
        return x

'''
    Calculate the shapes for sampling
'''
def calc_z_shapes(n_channel, input_size, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


'''
    Calculates model loss, given log probability and log determinant
'''
def calc_loss(log_p, logdet, image_size, n_bins):
    n_pixel = image_size * image_size * 3

    loss = -math.log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (math.log(2) * n_pixel)).mean(),
        (log_p / (math.log(2) * n_pixel)).mean(),
        (logdet / (math.log(2) * n_pixel)).mean(),
    )


'''
    GLOW MODEL TRAINING
    Implements the training for a GLOW model, lasting {n_iter}, 
    saving a checkpoint every 1k iterations

    Algorithm:

        Generate {n_samples} samples to extract examples
        For each iteration in {n_iter}
            Normalize and bin image
            Forward pass image
            Calculate loss
            Backpropagate loss

            Every {image_every} iterations
                Save {n_samples} images
            Every 1000 iterations
                Save model checkpoint
'''
def train_glow(model, optimizer, dataset, device, 
               n_bits=5, image_size = (64,64), n_samples = 20, 
               n_iter = 200000, temp = 0.7, image_every=100, n_blocks=4, 
               lr=1e-4):

    n_bins = 2.0 ** n_bits # Calculate number of bins

    # Create array of n_samples which will be constant as model trains
    # Every X interations, the images generated from these samples will be saved 
    z_sample = []
    z_shapes = calc_z_shapes(3, image_size[0], n_blocks)

    for z in z_shapes:
        # Sample at some temperature (0.7 suggested in original paper)
        z_new = torch.randn(n_samples, *z) * temp 
        z_sample.append(z_new.to(device))

    for i in (pbar := tqdm(range(n_iter))):
        image = next(dataset)
        image = image.to(device)

        image = image * 255

        if n_bits < 8:
            image = torch.floor(image / 2 ** (8 - n_bits))

        image = image / n_bins - 0.5

        if i == 0:
            with torch.no_grad():
                log_p, log_det, _ = model.module(
                    image + torch.rand_like(image) / n_bins
                )
                continue
        else:
            log_p, log_det, _ = model(image + torch.rand_like(image) / n_bins)
        
        log_det = log_det.mean()

        loss, log_p, log_det = calc_loss(log_p, log_det, image_size[0], n_bins)
        
        model.zero_grad()
        # Warm-up learning rate
        warmup_lr = lr
        optimizer.param_groups[0]["lr"] = warmup_lr
        loss.backward()
        optimizer.step()

        pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}"
            )
        breakpoint()
        # Save an image every X iterations
        if i % image_every == 0:
            with torch.no_grad():
                save_image(
                    model_single.reverse(z_sample).data,
                    f"sample/{str(i + 1).zfill(6)}.png",
                    normalize=True,
                    nrow=10,
                    value_range=(-0.5, 0.5),
                )
        # Save checkpoint every 1000 iterations
        if i % 1000 == 1:
            torch.save(model.module.state_dict(), f'checkpoint/nf_checkpoint_{i}.pt')
            torch.save(optimizer.state_dict(), f'checkpoint/nf_optimizer_{i}.pt')

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure

'''
    GLOW MODEL EVALUATION
    Evaluates a model checkpoint saved to {model_path}

    Algorithm:
        For each batch in test_loader
            Load real images
            Generate {batch_size} fake images
            Reverse image normalization
            Compute SSIM
            Compute InceptionScore
            Compute FID

        Return SSIM, InceptionScore, FID

'''
def eval_glow(model_path, in_channel, K, L, test_loader, device, temp=0.7):

    model_single = GlowModel(in_channel, K, L)

    print('Loading model state from file: ' + model_path)
    state_dict = torch.load(model_path, weights_only=True)

    # Load weights from model path
    model = nn.DataParallel(model_single)
    model.load_state_dict(state_dict, assign=True)
    model.to(device)

    model.eval() # Set for evaluation

    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    inception_score = InceptionScore().to(device)
    fid = FrechetInceptionDistance(feature=64).to(device)

    with torch.no_grad():
        for real_images in tqdm(test_loader):

            # Batch of real images
            real_images = real_images.to(device, non_blocking=True)

            batch_size = real_images.shape[0]

            z_shapes = calc_z_shapes(in_channel, real_images.shape[-1], L)
            
            # Batch of fake images
            z_fake = []
            for z in z_shapes:
                z_new = torch.randn(batch_size, *z) * temp
                z_fake.append(z_new.to(device))
            
            fake_images = model_single.reverse(z_fake).data

            # Reverse the normalization
            breakpoint()
            fake_images = fake_images * 0.5 + 0.5
            #real_images = real_images * 0.5 + 0.5

            # Update SSIM
            ssim.update(fake_images, real_images)

            real_images_uint8 = (real_images * 255).clamp(0, 255).to(torch.uint8)
            fake_images_uint8 = (fake_images * 255).clamp(0, 255).to(torch.uint8)
            
            # Update Inception Score
            inception_score.update(fake_images_uint8)

            # Update FID
            fid.update(real_images_uint8, real=True)
            fid.update(fake_images_uint8, real=False)

    return ssim.compute().cpu().item(), inception_score.compute()[0].cpu().item(), fid.compute().cpu().item()


import pickle as pkl            
import glob
import numpy as np


if __name__ == '__main__':

    # Model and dataset hyperparameters
    n_bits = 5              # Number of bits
    image_size = (64, 64)   # Image size (pixels)
    n_samples = 20          # Number of samples (examples)
    n_iter = 30000          # Number of training iterations
    temp = 0.7              # Sampling temperature  
    K = 32                  # Number of flows per block
    L = 4                   # Number of blocks
    lr = 1e-4               # Learning rate
    batch_size = 16         # Batch size
    image_every = 100       # Print samples every X images
    num_workers = 4         # DataLoader num workers

    training = True        # Whether to perform training

    # Initialize torch on cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    if training:
        model_single = GlowModel(3, K, L)
        model = nn.DataParallel(model_single)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Get training dataset
        train_dataset, _, _ = get_dataset_loaders(r'C:\Users\rodri\celeba\raw', 
                                                         batch_size=batch_size,
                                                         dataset_type='LQ',
                                                         num_workers=num_workers)

        train_glow(model, optimizer, iter(train_dataset), device,
                n_bits=n_bits,
                image_size=image_size,
                n_samples=n_samples,
                n_iter=n_iter,
                temp=temp,
                image_every=image_every,
                n_blocks=L,
                lr=lr)
    
    # Get testing dataset
    _, test_dataset, _ = get_dataset_loaders(r'C:\Users\rodri\celeba\raw', 
                                                         batch_size=batch_size,
                                                         dataset_type='LQ')

    # Evaluation - select latest checkpoint
    checkpoints = glob.glob('checkpoint/*')
    check_path = checkpoints[np.argmax([
        int(c.removeprefix('checkpoint\\nf_checkpoint_').removesuffix('.pt')) for c in checkpoints])]
    ssim, inception, fid = eval_glow(check_path, 3, K, L, test_dataset, device)
    
    # Print evaluation results
    print(f'Evaluation results:\n SSIM: {ssim}, Inception Score: {inception}, FID: {fid}')
    
    # Saves Metrics to a pickle binary file
    with open('nf_results.pkl', 'wb') as handle:
        pkl.dump({'SSIM': ssim, 'INC': inception, 'FID':fid}, handle)