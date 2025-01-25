import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class TactileMaterialDataset(Dataset):
    def __init__(self, file_path, split='train', train_split=0.8, sequence_length=1000):
        self.sequence_length = sequence_length
        
        with h5py.File(file_path, 'r') as dataset:
            samples = dataset['samples'][:]
            materials = dataset['materials'][:]
            materials = [m.decode('utf-8') for m in materials]

        np.random.seed(42)

        # Shuffle samples
        for i in range(samples.shape[0]):
            np.random.shuffle(samples[i])

        total_size = samples.shape[1]
        train_size = int(train_split * total_size)
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        if split == 'train':
            chosen_indices = indices[:train_size]
        elif split == 'val':
            chosen_indices = indices[train_size:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        data_split = samples[:, chosen_indices, ...]  
        label_split = np.repeat(np.arange(samples.shape[0]), len(chosen_indices))

        # Flatten
        self.full_sequences = data_split.reshape(-1, data_split.shape[2], data_split.shape[3], data_split.shape[4])
        self.labels = label_split

        all_snippets = []
        all_labels = []

        for i in range(len(self.full_sequences)):
            full_seq = self.full_sequences[i]  # shape: [T, 4, 4]
            label = self.labels[i]

            T = full_seq.shape[0]
            if T < self.sequence_length:
                raise ValueError(f"Sequence length {self.sequence_length} is longer than original sequence length {T}")

            start_idx = np.random.randint(0, T - self.sequence_length + 1)
            snippet = full_seq[start_idx : start_idx + self.sequence_length]

            # Flatten [4,4] -> 16
            snippet = snippet.reshape(self.sequence_length, 16)

            all_snippets.append(snippet)
            all_labels.append(label)

        self.snippets = torch.tensor(all_snippets, dtype=torch.float32)
        self.labels   = torch.tensor(all_labels, dtype=torch.long)

        # Add the channel dimension if your model expects [N, 1, T, 16]
        self.snippets = self.snippets.unsqueeze(1)  # shape => [N, 1, seq_len, 16]

        print(f"{split.capitalize()} dataset final size: {len(self.snippets)}")
        print(f"Snippet shape: {self.snippets.shape}")  

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        return self.snippets[idx], self.labels[idx]

