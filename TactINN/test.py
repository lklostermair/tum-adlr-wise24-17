import sys
import os

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from torch.utils.data import DataLoader
from utils import TactileMaterialDataset
from tqdm import tqdm


class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.net(x)
    
class SimpleCNN1D(nn.Module):
    """
    A small 1D CNN that processes [B, 16, T] and outputs scale + shift for an (16, T)-shaped input.
    """
    def __init__(self,
                 in_channels=16,       # 16 taxel channels
                 hidden_channels=16,
                 kernel_size=3,
                 length=1000,         # Temporal length
                 label_dim=0):
        super(SimpleCNN1D, self).__init__()
        self.label_dim = label_dim
        self.in_channels = in_channels
        self.length = length

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=self.in_channels + (1 if label_dim > 0 else 0),
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)  # Halves the temporal length
        self.relu = nn.ReLU()

        # Flattened dimension after pooling
        reduced_length = self.length // 2
        self.flatten_dim = hidden_channels * reduced_length

        # Output scale and shift
        self.fc_out = nn.Linear(self.flatten_dim, 2 * in_channels * self.length)

    def forward(self, x, y=None):
        """
        x: [B, 16, T] - Input tensor
        y: [B, label_dim] - Optional label embedding
        """
        if y is not None:
            # Broadcast label embedding to match input temporal length
            y = y.unsqueeze(-1).expand(-1, -1, x.size(-1))  # [B, label_dim, T]
            x = torch.cat([x, y], dim=1)  # [B, in_channels + label_dim, T]

        x = self.conv1(x)  # [B, hidden_channels, T]
        x = self.relu(x)
        x = self.conv2(x)  # [B, hidden_channels, T]
        x = self.relu(x)
        x = self.pool(x)   # [B, hidden_channels, T/2]

        # Flatten and compute scale + shift
        x = x.flatten(1)  # [B, hidden_channels * (T/2)]
        x = self.fc_out(x)  # [B, 2 * (16 * T)]
        scale, shift = torch.chunk(x, 2, dim=1)  # Each => [B, 16 * T]
        return scale, shift



def affine_coupling_block(in_channels, hidden_dim=64):
    def subnet_constructor(c_in, c_out):
        return SimpleCNN1D(in_channels, hidden_dim, label_dim=0)

    return Fm.AllInOneBlock(
        dims_in=[(in_channels,)],
        subnet_constructor=subnet_constructor,
        affine_clamping=2.0,
        permute_soft=True,
    )

class InvertibleFlow(nn.Module):
    def __init__(self, input_dim, n_blocks=4, hidden_dim=64):
        super(InvertibleFlow, self).__init__()
        self.flow = Ff.SequenceINN(input_dim)
        for _ in range(n_blocks):
            self.flow.append(affine_coupling_block(input_dim, hidden_dim))

    def forward(self, x):
        z, log_detJ = self.flow(x)
        return z, log_detJ

    def inverse(self, z):
        x, log_detJ = self.flow(z, rev=True)
        return x, log_detJ


class ClassConditionalGMM(nn.Module):
    def __init__(self, n_classes, latent_dim):
        super(ClassConditionalGMM, self).__init__()
        self.mu = nn.Parameter(torch.zeros(n_classes, latent_dim))
        self.logvar = nn.Parameter(torch.zeros(n_classes, latent_dim))

    def forward(self, z):
        # Compute log-probabilities for each class
        z_expanded = z.unsqueeze(1)
        mu = self.mu.unsqueeze(0)
        logvar = self.logvar.unsqueeze(0)
        log_p = -0.5 * (logvar + (z_expanded - mu)**2 / logvar.exp())
        return log_p.sum(dim=-1)

def combined_loss(x, c, model, gmm, beta=1.0):
    z, log_detJ = model(x)
    log_pz = -0.5 * (z**2).sum(dim=1)
    L_gen = -(log_pz + log_detJ).mean()

    log_pc = gmm(z)
    L_cls = nn.CrossEntropyLoss()(log_pc, c)
    return L_gen + beta * L_cls



def train_model(h5_file, epochs=10, batch_size=32, lr=1e-3):
    # Dataset & DataLoader
    train_set = TactileMaterialDataset(h5_file, split='train')
    val_set = TactileMaterialDataset(h5_file, split='val')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Model and Optimizer
    input_dim = 16 * 1000
    model = InvertibleFlow(input_dim, n_blocks=4, hidden_dim=64)
    gmm = ClassConditionalGMM(n_classes=36, latent_dim=input_dim)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(gmm.parameters()), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training Phase
        model.train()
        gmm.train()
        train_loss = 0

        with tqdm(total=len(train_loader), desc="Training", leave=False) as pbar:
            for x_batch, c_batch in train_loader:
                x_batch = x_batch.view(x_batch.size(0), -1)  # Flatten [B, T, C] -> [B, T*C]
                optimizer.zero_grad()
                loss = combined_loss(x_batch, c_batch, model, gmm)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        print(f"  Training Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        gmm.eval()
        val_loss, correct, total = 0, 0, 0

        with tqdm(total=len(val_loader), desc="Validation", leave=False) as pbar:
            for x_batch, c_batch in val_loader:
                x_batch = x_batch.view(x_batch.size(0), -1)
                with torch.no_grad():
                    loss = combined_loss(x_batch, c_batch, model, gmm)
                    val_loss += loss.item()

                    z, _ = model(x_batch)
                    preds = gmm(z).argmax(dim=1)
                    correct += (preds == c_batch).sum().item()
                    total += c_batch.size(0)
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        print(f"  Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}\n")


if __name__ == "__main__":
    train_model(h5_file="/home/luki/tum-adlr-wise24-17/data/raw/tactmat.h5", epochs=10, batch_size=16, lr=1e-3)
