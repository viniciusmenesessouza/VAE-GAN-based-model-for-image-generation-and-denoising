import numpy as np
import glob
from scipy.signal import decimate
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

class EEGLoad():
    def __init__(self, data, downsampling = 4, event_type = 'mi'):
        self.fs = 250
        self.data = data['s'].T
        self.data = self.data[0:22]
        self.events_type = data['etyp'].T
        self.events_pos = data['epos'].T
        self.downsampling = downsampling
        self.get_type = event_type

    def get_trials(self):
        if self.get_type == 'mi':
            start_trial = [769, 770, 771, 772, 783] # Mi cue onset
            start_events = [True if event in start_trial else False for event in self.events_type[0]]
            idxs = [i for i,x in enumerate(start_events) if x]
        else: # get baseline
            start_trial = 768 # start of trial (beep)
            start_events = self.events_type == start_trial
            idxs = [i for i, x in enumerate(start_events[0]) if x]

        trials = []
        for idx in idxs:
            try:
                start = self.events_pos[0, idx]
                stop = int(start + 640)
                trial = self.data[:, start:stop]
                trial = decimate(trial, self.downsampling)
                trials.append(trial)
            except:
                continue
        
        return trials

class EEGDataset():
    def __init__(self, path='eeg_data/*', dataset_type = 'T', event_type='mi', subject=None):

        path += dataset_type + '.npz'

        data_paths = glob.glob(path)
        
        if subject:
            data_paths = [path for path in data_paths if (subject in path)]

        all_trials = []
        for path in data_paths:

            data = EEGLoad(np.load(path), event_type = event_type).get_trials()
            #print(f'Loaded data from file {path}, total of {len(data)} trials')
            
            all_trials.append(data)
        
        dataset = []
        for subject_trials in all_trials:
            for trial in subject_trials:
                dataset.append(trial)
        self.dataset = np.array(dataset)

    def __getitem__(self, index):
        return torch.Tensor(self.dataset[index]).unsqueeze(2)

    def __len__(self):
        return len(self.dataset)


from nf_code import ActivationNormalization, gaussian_log_p, gaussian_sample, calc_loss, InvertibleConv

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

        self.conv = nn.Conv2d(in_channel, out_channel, (3,1), padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        # Note: don't pad last dimension
        out = nn.functional.pad(input, [0, 0, 1, 1], value=1)

        out = self.conv(out)

        out = out * torch.exp(self.scale * 3)

        return out

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
            nn.Conv2d(in_channel // 2, filter_size, (ksize,1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, (1,1)),
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
        self.in_size = in_channel
        self.flows = nn.ModuleList()
        for _ in range(K):
            self.flows.append(Flow(self.in_size * 2))
        
        # Whether to split at the end of the block
        self.split = split
        
        if split:
            self.prior = ZeroConv2d(self.in_size, self.in_size * 2)
        else:
            self.prior = ZeroConv2d(self.in_size * 2, self.in_size * 4)
        

    def forward(self, x):
        batch_size, n_channels, h, w = x.shape
        
        #squeeze = x.view(batch_size, n_channels, h // 2, 2, w // 2, 2)
        squeeze = x.view(batch_size, n_channels, h // 2 , 2, 1, 1)
        squeeze = squeeze.permute(0, 1, 3, 5, 2, 4)
        
        #y = squeeze.contiguous().view(batch_size, n_channels * 4, h // 2, 1)
        y = squeeze.contiguous().view(batch_size, n_channels * 2, h // 2, 1)
        
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
        

        batch_size, num_channels, h, w = x.shape
        
        unsqueeze = x.view(batch_size, num_channels // 2, 2, 1, h, 1)
        
        unsqueeze = unsqueeze.permute(0, 1, 4, 2, 5, 3)
        
        unsqueeze = unsqueeze.contiguous().view(
            batch_size, num_channels // 2, h * 2, 1
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


from tqdm import tqdm
import mne
import matplotlib

def train_glow(model, optimizer, dataset, device, 
               num_chans = 22, n_samples = 16, 
               n_iter = 200000, temp = 0.7, image_every=1, 
               lr=1e-4, save_name = ''):

    channel_names = ['Fz', 
                     'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                     'C5', 'C3', 'C1', 'Cz', 'C2','C4','C6',
                     'CP3', 'CP1','CPz','CP2','CP4',
                     'P1','Pz','P2',
                     'POz']
    info = mne.create_info(channel_names, sfreq = 250/4)

    # Create array of n_samples which will be constant as model trains
    # Every X interations, the images generated from these samples will be saved 
    z_sample = []
    
    z_shapes = [(num_chans, 80, 1),
                (num_chans, 40, 1), 
                (num_chans, 20, 1), 
                (num_chans * 2, 10, 1)]
    
    for z in z_shapes:
        # Sample at some temperature (0.7 suggested in original paper)
        z_new = torch.randn(n_samples, *z) * temp 
        z_sample.append(z_new.to(device))
    
    iterator = iter(dataset)
    
    for i in (pbar := tqdm(range(n_iter + 1))):
        
        try:
            x = next(iterator)
        except StopIteration:
            iterator = iter(dataset)
            x = next(iterator)

        x = x.to(device)
        if i == 0:
            with torch.no_grad():
                log_p, log_det, _ = model(
                    x + torch.rand_like(x)
                )
                continue
        else:
            log_p, log_det, _ = model(x + torch.rand_like(x))
        
        log_det = log_det.mean()

        loss, log_p, log_det = calc_loss(log_p, log_det, x.shape[2], 1)
        
        model.zero_grad()
        # Warm-up learning rate
        warmup_lr = lr
        optimizer.param_groups[0]["lr"] = warmup_lr
        loss.backward()
        optimizer.step()

        pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}"
            )

        # Sample the fake EEG signal
        if i % image_every == 0:
            
            fakes = model.reverse(z_sample).data
            fakes = fakes.squeeze().detach().cpu().numpy()
            
            # Save the fake signal for comparisons later
            np.savez('sample/fake_eeg_' + save_name + f'_{i}', fakes)
            
            # Create and save image
            epoch = mne.io.RawArray(fakes[0], info)
            fig = epoch.plot(show_scrollbars=False, show_scalebars=False, verbose=False)
            fig.savefig('sample/fake_eeg_' + save_name + f'_{i}')

        # Save a checkpoint
        if i % 2000 == 0:
            torch.save(model.state_dict(), 'checkpoint/eeg_flow_model_' + save_name + f'_{i}.pt')
            torch.save(optimizer.state_dict(), 'checkpoint/eeg_flow_optimizer_' + save_name + f'_{i}.pt')


if  __name__ == '__main__':
    
    matplotlib.use('Agg')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This will train a model for each type of epoch (MI and Baseline)
    #   Note that this could also be done with conditioning the output,
    #   in which case only one model would be necessary
    
    print('Starting training loop. Please make sure you have enough storage space.')
    print('Every 2000 training iterations model and optimizer states will be saved, taking up roughly 600 MB.')
    
    etypes = ['mi', 'baseline']
    #etypes = ['baseline']
    
    for event_type in etypes:
        
        print(f'Training model for event type: {event_type}')
        # Instantiate the adapted Glow Model
        model = GlowModel(22, 32, 4)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters())

        
        # Load training datasets, MI / baseline events
        eeg_dataset = EEGDataset(dataset_type='T', event_type = event_type)

        train_loader = DataLoader(eeg_dataset, 16)
        
        # Main model training loop, refer to documentation
        train_glow(model, 
                optimizer, 
                train_loader, 
                device,
                n_samples=16,
                n_iter=6000,
                temp=0.7,
                image_every=1000,
                save_name= event_type + '_training'
                )
