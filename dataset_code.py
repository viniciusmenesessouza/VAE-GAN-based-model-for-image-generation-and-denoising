import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pickle

class CelebADatasetV0(Dataset):
    def __init__(self, image_folder, image_size, noise_max_std=None, rect=True, dataset_size=None):
        # noise_max_std: if None, the dataset return just clean images
        # dataset_size: if None, the dataset has all images from the given folder
        self.image_folder = image_folder
        pattern = os.path.join(image_folder, '*.jpg')
        self.image_paths = list(glob.iglob(pattern, recursive=False))
        if dataset_size is not None:
            self.image_paths = self.image_paths[:dataset_size]
        self.image_size = image_size
        self.noise_max_std = noise_max_std
        self.rect = rect

        self.clean_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def add_noise(self, img):
        noise = torch.randn_like(img) * torch.rand(1)*self.noise_max_std
        if self.rect:
            noisy_img = self.add_random_rectangle(img) + noise
        else:
            noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, -1.0, 1.0)
        return noisy_img

    def add_random_rectangle(self, img, min_size=0.01, max_size=0.25):
        channels, H, W = img.shape
        color = torch.rand(channels)*2.0-1.0
        min_size = round(min(H,W)*min_size)
        max_size = round(min(H,W)*max_size)
        rect_h = torch.randint(min_size, max_size + 1, (1,)).item()
        rect_w = torch.randint(min_size, max_size + 1, (1,)).item()
        x = torch.randint(round(W*0.25), round(W*0.75)+1-rect_w, (1,)).item()
        y = torch.randint(round(H*0.25), round(H*0.75)+1-rect_h, (1,)).item()
        img_with_rect = img.clone()
        for c in range(channels):
            img_with_rect[c, y:y + rect_h, x:x + rect_w] =  torch.rand((rect_h, rect_w)) * 2.0 - 1.0 # color[c]
        return img_with_rect

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = default_loader(image_path)
        clean_img = self.clean_transform(image)
        if self.noise_max_std is None:
            return clean_img
        noisy_img = self.add_noise(clean_img)
        return noisy_img, clean_img

def _load_and_transform(args):
    path, transform = args
    image = default_loader(path)
    return transform(image)

class CelebADataset(Dataset):
    def __init__(self, image_folder, image_size, noise_max_std=None, rect=True, dataset_size=None):
        # noise_max_std: if None, the dataset return just clean images
        # dataset_size: if None, the dataset has all images from the given folder
        self.image_folder = image_folder
        pattern = os.path.join(image_folder, '*.jpg')
        self.image_paths = list(glob.iglob(pattern, recursive=False))
        if dataset_size is not None:
            self.image_paths = self.image_paths[:dataset_size]
        self.image_size = image_size
        self.noise_max_std = noise_max_std
        self.rect = rect

        self.clean_transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # self.cached_data = []
        # for image_path in self.image_paths:
        #     image = default_loader(image_path)
        #     clean_img = self.clean_transform(image)
        #     self.cached_data.append(clean_img)
        print(f'Loading {len(self.image_paths)} images...')
        with Pool(cpu_count() - 2) as pool:
            args = []
            for i in range(len(self.image_paths)):
                args.append((self.image_paths[i], self.clean_transform))
            self.cached_data = list(tqdm(pool.imap(_load_and_transform, args), total=len(args)))

    def __len__(self):
        return len(self.image_paths)

    def add_noise(self, img):
        noise = torch.randn_like(img) * torch.rand(1)*self.noise_max_std
        if self.rect:
            noisy_img = self.add_random_rectangle(img) + noise
        else:
            noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, -1.0, 1.0)
        return noisy_img

    def add_random_rectangle(self, img, min_size=0.01, max_size=0.25):
        channels, H, W = img.shape
        color = torch.rand(channels)*2.0-1.0
        min_size = round(min(H,W)*min_size)
        max_size = round(min(H,W)*max_size)
        rect_h = torch.randint(min_size, max_size + 1, (1,)).item()
        rect_w = torch.randint(min_size, max_size + 1, (1,)).item()
        x = torch.randint(round(W*0.25), round(W*0.75)+1-rect_w, (1,)).item()
        y = torch.randint(round(H*0.25), round(H*0.75)+1-rect_h, (1,)).item()
        img_with_rect = img.clone()
        for c in range(channels):
            img_with_rect[c, y:y + rect_h, x:x + rect_w] =  torch.rand((rect_h, rect_w)) * 2.0 - 1.0 # color[c]
        return img_with_rect

    def __getitem__(self, idx):
        clean_img = self.cached_data[idx]
        if self.noise_max_std is None:
            return clean_img
        noisy_img = self.add_noise(clean_img)
        return noisy_img, clean_img

def get_dataset_loaders(path, batch_size=64, train_p=0.8, val_p=0.1, image_size=(218,178), noise_max_std=0.1, rect=True, dataset_size=None, pickle_path='dataset_gan_obj.pkl'):
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = CelebADataset(path, image_size, noise_max_std, rect, dataset_size)
        with open(pickle_path, "wb") as f:
            pickle.dump(dataset, f)
    dataset_size = len(dataset)
    train_size = round(train_p * dataset_size)
    val_size = round(val_p * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.dataset.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # with open('testloader_gan_obj.pkl', "wb") as f:
    #     pickle.dump(test_loader, f)
    return train_loader, val_loader, test_loader, dataset[0][0].numpy().shape