import numpy as np
import torch
import h5py
from torchvision.transforms import transforms as T

class Tactnet():
    def __init__(self, batch_size):
        super().__init__()

        with h5py.File('data/raw/tactmat.h5', 'r') as dataset:
            samples = dataset['samples'][:].reshape(-1, 4, 4000)[:,:,:3969].reshape(-1, 4, 63, 63)
            materials = [m.decode('utf-8') for m in dataset['materials'][:]]
            
        labels = np.repeat(range(len(materials)), 100)  # 100 samples per material

        indices = np.arange(len(labels))
        np.random.shuffle(indices)

        # Split indices into train and test
        train_size = int(0.8 * len(indices))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_samples, val_samples = samples[train_indices]/154.0, samples[val_indices]/154.0
        train_labels, val_labels = labels[train_indices], labels[val_indices]

        self.val_fast_set_size = 50  # Use only 50 randomly picked files for validation steps performed within epochs

        self.batch_size = batch_size

        self.n_classes = 36
        self.img_crop_size = (64, 64)

        # These are calculated from train after resizing to 64x64
        self._mu_img = [0.0698, 0.0558, 0.0505, 0.0477]
        self._std_img = [0.0277, 0.0195, 0.0168, 0.0156]

        self._all_one_hot_encodings = torch.eye(self.n_classes)

        self.train_data = TactnetDataset(train_samples, train_labels, transform=T.Compose([
            T.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            T.RandomResizedCrop(self.img_crop_size),
            T.RandomHorizontalFlip(),
            T.Lambda(self._uniform_noise),
            T.Normalize(self._mu_img, self._std_img)
        ]), target_transform=T.Compose([T.Lambda(self._class_to_soft_hot)]))

        self.val_data_fast = TactnetDataset(val_samples, val_labels, transform=T.Compose([
            T.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            T.Resize(self.img_crop_size),
            T.CenterCrop(self.img_crop_size),
            T.Normalize(self._mu_img, self._std_img)
        ]), target_transform=T.Compose([T.Lambda(self._class_to_one_hot)]))

        indices = np.random.choice(len(val_samples), self.val_fast_set_size).tolist()
        self.val_data_fast.samples = [val_samples[i] for i in indices]
        self.val_data_fast.labels = [val_labels[i] for i in indices]
        
        self.val_data = TactnetDataset(val_samples, val_labels, transform=T.Compose([
            T.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            T.Resize(self.img_crop_size),
            T.CenterCrop(self.img_crop_size),
            T.Normalize(self._mu_img, self._std_img),
        ]), target_transform=T.Compose([T.Lambda(self._class_to_one_hot)]))

        self.val_data_10_crop = TactnetDataset(val_samples, val_labels, transform=T.Compose([
            T.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            T.Resize(self.img_crop_size),
            T.TenCrop(self.img_crop_size),
            T.Lambda(lambda crops: torch.stack([T.Normalize(self._mu_img, self._std_img)(crop) for crop in crops])),
        ]), target_transform=T.Compose([T.Lambda(self._class_to_one_hot)]))

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=32, shuffle=True, num_workers=0, pin_memory=False, sampler=None)
        self.val_loader_fast = torch.utils.data.DataLoader(self.val_data_fast, batch_size=32, shuffle=False, num_workers=0, pin_memory=False, sampler=None)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=32, shuffle=True, num_workers=0, pin_memory=False, sampler=None)
        self.val_loader_10_crop = torch.utils.data.DataLoader(self.val_data_10_crop, batch_size=32, shuffle=True, num_workers=0, pin_memory=False, sampler=None)

    def set_model(self, model):
        self.train_loader.set_model(model)

    def _uniform_noise(self, x):
        return torch.clamp(x + torch.rand_like(x) / 154., min=0., max=1.)

    def _class_to_soft_hot(self, y):
        hard_one_hot = self._all_one_hot_encodings[y]
        return hard_one_hot * (1 - 0.05) + 0.05 / self.n_classes

    def _class_to_one_hot(self, y):
        hard_one_hot = self._all_one_hot_encodings[y]
        return hard_one_hot

class TactnetDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, transform=None, target_transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            label = self.target_transform(label)

        return sample, label
