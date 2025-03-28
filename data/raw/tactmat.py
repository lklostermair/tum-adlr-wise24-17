import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import h5py


class tactileDataset(Dataset):
    def __init__(self, h5_file, transform=None):
        with h5py.File(h5_file, 'r') as dataset:
            self.samples = dataset['samples'][:].reshape(-1, 1000, 4, 4)
            self.materials = [m.decode('utf-8') for m in dataset['materials'][:]]
        
        self.transform = transform
        self.labels = np.repeat(range(len(self.materials)), 100)  # 100 samples per material

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


class tactileDataModule:
    def __init__(self, h5_file, batch_size, num_classes=36, k_folds=5):
        self.h5_file = h5_file
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.k_folds = k_folds

        self.dataset = tactileDataset(h5_file)
        self.sampler_train_test = StratifiedKFold(n_splits=k_folds, shuffle=True)
        self.sampler_train_val = StratifiedKFold(n_splits=k_folds, shuffle=True)

    def get_data_loaders(self):
        indices = np.arange(len(self.dataset))
        labels = self.dataset.labels

        train_indices, test_indices = self.sampler.split(indices, labels)
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(train_indices))
        test_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(test_indices))

        return train_loader, test_loader