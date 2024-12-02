import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TactileMaterialDataset(Dataset):
    def __init__(self, file_path, split='train', train_split=0.8):
        """
        Custom Dataset for loading and processing tactile material data.

        Args:
            file_path (str): Path to the HDF5 file.
            split (str): Dataset split, either 'train' or 'val'.
            train_split (float): Proportion of the dataset to use for training.
        """
        with h5py.File(file_path, 'r') as dataset:
            samples = dataset['samples'][:]  # Shape: [materials, samples, time_steps, taxels_x, taxels_y]
            materials = dataset['materials'][:]
            materials = [m.decode('utf-8') for m in materials]  # Decode material names if necessary

        # Shuffle samples within each class
        for i in range(samples.shape[0]):
            np.random.shuffle(samples[i])

        # Splits dataset into train and validation
        total_size = samples.shape[1]  # Number of samples per class
        train_size = int(train_split * total_size)

        if split == 'train':
            self.data = samples[:, :train_size, ...]  # Training samples
            self.labels = np.repeat(np.arange(samples.shape[0]), train_size)  # Class labels
        elif split == 'val':
            self.data = samples[:, train_size:, ...]  # Validation samples
            self.labels = np.repeat(np.arange(samples.shape[0]), total_size - train_size)
        else:
            raise ValueError("Invalid split value. Must be 'train' or 'val'.")

        # Flatten the taxels (4x4 -> 16) and reshape to [num_samples, time_steps, features]
        self.data = self.data.reshape((-1, samples.shape[2], 16))

        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



