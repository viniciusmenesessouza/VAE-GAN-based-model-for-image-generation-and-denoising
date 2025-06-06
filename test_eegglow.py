import matplotlib.pyplot as plt
import imageio.v3 as iio
import glob
import numpy as np
from scipy.signal import welch
from flow_eeg import EEGDataset
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
import pickle as pkl



def plot_sample_gifs():
    for image_type in ['mi', 'baseline']:

        gif_path = f'sample/fake_eeg_{image_type}_training.gif'
        
        if glob.glob(gif_path):
            continue
        images = glob.glob(f'sample/fake_eeg_{image_type}_training_*.png')
        
        frames = np.stack([iio.imread(img_path) for img_path in images])
        iio.imwrite(gif_path, frames, duration=0.75)


def plot_sample_psd():
    plt.figure()
    for sample_type in ['mi', 'baseline']:
        sample_path = glob.glob(f'sample/fake_eeg_{sample_type}_training_*.npz')
        sample_path = sample_path[np.argmax(
            [int(path.removeprefix(f'sample\\fake_eeg_{sample_type}_training_').removesuffix('.npz')) 
             for path in sample_path])]
        
        samples = np.load(sample_path)['arr_0']
        # Create a surrogate channel that averages C3 (7), Cz(9), C4(11)
        samples = (samples[:,7,:] + samples[:,9,:] + samples[:,11,:])/3
        psds = []
        for sample in samples:
            fx, psd = welch(sample, fs=250/4, nperseg=64)
            psds.append(psd)

        plt.plot(fx, np.mean(psds, axis=0), label=sample_type)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power [V**2/Hz]')
    plt.legend()
    plt.xlim([0, 30])
    plt.savefig('sample/fake_eeg_psd.png')


from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from flow_eeg import GlowModel
import torch
from scipy.signal import butter, lfilter

def get_samples(device, sample_type='mi', n_samples=200, num_chans=22):
    model = GlowModel(num_chans, 32, 4)
    state_dict = torch.load(f'checkpoint/eeg_flow_model_{sample_type}_training_6000.pt')
    model.load_state_dict(state_dict)
    model.to(device)
    
    model.eval()

    # Create array of n_samples which will be constant as model trains
    # Every X interations, the images generated from these samples will be saved 
    z_sample = []
    
    z_shapes = [(num_chans, 80, 1),
                (num_chans, 40, 1), 
                (num_chans, 20, 1), 
                (num_chans * 2, 10, 1)]
    
    for z in z_shapes:
        # Sample at some temperature (0.7 suggested in original paper)
        z_new = torch.randn(n_samples, *z) * 0.7
        z_sample.append(z_new.to(device))
    
    return model.reverse(z_sample).data

def plot_eeg_psd(subject = '01'):
    epochs_mi = EEGDataset(event_type='mi', subject=subject).dataset
    epochs_baseline = EEGDataset(event_type='baseline', subject=subject).dataset
    
    plt.figure()
    for i, epochs in enumerate([epochs_mi, epochs_baseline]):
        epochs = (epochs[:,7,:] + epochs[:,9,:] + epochs[:,11,:])/3
        psds = []
        for epoch in epochs:
            f, psd = welch(epoch, fs=250/4, nperseg=64)
            psds.append(psd)
        plt.plot(f, np.mean(psds, axis=0), label=['mi','baseline'][i])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power [V**2/Hz]')
    plt.legend()
    plt.xlim([0, 30])
    plt.savefig(f'sample/real_eeg_psd_{subject}.png')

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == '__main__':

    subjects = ['01','02','03','04','05','06','07','08','09']

    # Plot sample PSDs
    plot_sample_psd()
    for subject in subjects:
        plot_eeg_psd(subject)
    # Generate sample gifs
    plot_sample_gifs()
    methods = ['synt', 'real', 'sr', 'other_sub']
    methods = ['other_sub']
    
    for method in methods:
        # Save train, test accuracy, f1score, kappa
        train_results = {s:[] for s in subjects}
        test_results = {s:[] for s in subjects}
        # Loading testing data
        for i, subject in enumerate(subjects):
            
            X_test_mi = EEGDataset(dataset_type='E', event_type='mi', subject=subject).dataset
            X_test_baseline = EEGDataset(dataset_type='E', event_type='baseline', subject=subject).dataset
            
            # Take test data from subject (always real EEG)
            X_test = np.concatenate((X_test_mi, X_test_baseline)).astype(np.float64)
            Y_test = np.concatenate((np.ones(len(X_test_mi)), np.zeros(len(X_test_baseline))))

            
            # Train classifier on just synthetic data
            if method == 'synt' or method == 'sr':
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
                
                X_mi = get_samples(device, sample_type='mi', n_samples=288)
                X_baseline = get_samples(device, sample_type='baseline', n_samples=288)

                X_train_synt = torch.cat((X_mi, X_baseline)).detach().cpu().numpy().squeeze().astype(np.float64)
                Y_train_synt = np.concatenate((np.ones(len(X_mi)), np.zeros(len(X_baseline))))

            # Train classifier on just real data
            if method == 'real' or method == 'sr':
                X_mi = EEGDataset(dataset_type='T', event_type='mi', subject=subject).dataset
                X_baseline =  EEGDataset(dataset_type='T', event_type='baseline', subject=subject).dataset   
                X_train_real = np.concatenate((X_mi, X_baseline)).astype(np.float64)
                Y_train_real = np.concatenate((np.ones(len(X_mi)), np.zeros(len(X_baseline))))
            
            if method == 'other_sub':
                other_sub = subjects[i-1]
                X_mi = EEGDataset(dataset_type='T', event_type='mi', subject=other_sub).dataset
                X_baseline =  EEGDataset(dataset_type='T', event_type='baseline', subject=other_sub).dataset   
                X_train_real = np.concatenate((X_mi, X_baseline)).astype(np.float64)
                Y_train_real = np.concatenate((np.ones(len(X_mi)), np.zeros(len(X_baseline))))
            

            if method == 'sr':
                X_train = np.concatenate((X_train_synt, X_train_real))
                Y_train = np.concatenate((Y_train_synt, Y_train_real))
            elif method == 'synt':
                X_train = X_train_synt
                Y_train = Y_train_synt
            elif method == 'real' or method == 'other_sub':
                X_train = X_train_real
                Y_train = Y_train_real


            # Apply a simple BP + CSP + LDA
            lda = LinearDiscriminantAnalysis()
            csp = CSP()
            
            X_train = butter_bandpass_filter(X_train, 8, 30, 250/4)

            X_train = csp.fit_transform(X_train, Y_train)
            lda.fit(X_train, Y_train)
            Y_pred_train = lda.predict(X_train)

            train_results[subject] = {'acc':accuracy_score(Y_train, Y_pred_train), 
                                    'f1': f1_score(Y_train, Y_pred_train),
                                    'k':cohen_kappa_score(Y_train, Y_pred_train) }
            
            print(f'Subject {subject}, method {method}')
            print(f'Training results: {train_results[subject]["acc"]} accuracy, {train_results[subject]["f1"]} f1-score, {train_results[subject]["k"]} kappa')
            
            X_test = butter_bandpass_filter(X_test, 8, 30, 250/4)
            X_test = csp.transform(X_test)
            Y_pred_test = lda.predict(X_test)

            test_results[subject] =  {'acc':accuracy_score(Y_test, Y_pred_test), 
                                    'f1': f1_score(Y_test, Y_pred_test),
                                    'k':cohen_kappa_score(Y_test, Y_pred_test) }
            
            print(f'Testing results: {test_results[subject]["acc"]} accuracy, {test_results[subject]["f1"]} f1-score, {test_results[subject]["k"]} kappa')

        pkl.dump(train_results, open(f'sample/train_res_{method}.pkl', 'wb'))
        pkl.dump(test_results, open(f'sample/test_res_{method}.pkl', 'wb'))






