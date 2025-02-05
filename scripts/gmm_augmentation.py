from sklearn.mixture import GaussianMixture
import numpy as np
import os
import sys
import h5py
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data_loader import TactileMaterialDataset
from models.TactNetII_model import TactNetII

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_dir = "output"  
os.makedirs(output_dir, exist_ok=True)

def train_gmms(data, num_materials, n_components=5):
    """
    Train a GMM for each material.

    Args:
        data: The dataset as a NumPy array of shape [materials, samples, timesteps, taxels].
        num_materials: Number of materials in the dataset.
        n_components: Number of GMM components.

    Returns:
        A list of trained GMM models for each material.
    """
    gmms = []

    for material_idx in range(num_materials):
        # Extract all samples for this material
        material_data = data[material_idx]  # Shape: [samples, timesteps, taxels_x, taxels_y]
        
        # Extract the last 500 timesteps and flatten spatial dimensions
        last_500_data = material_data[:, -500:, :, :].reshape(-1, 500 * 16)  # Shape: [samples, 500 * 16]
        
        # Train GMM
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(last_500_data)
        gmms.append(gmm)
        print(f"Trained GMM for material {material_idx + 1}/{num_materials}")

    return gmms

def generate_synthetic_samples(data, gmms, sequence_length=1000, synthetic_length=500):
    """
    Generate synthetic samples for the dataset using trained GMMs.

    Args:
        data: Original dataset as a NumPy array of shape [materials, samples, timesteps, taxels].
        gmms: List of trained GMM models for each material.
        sequence_length: Original sequence length.
        synthetic_length: Length of synthetic data to append.

    Returns:
        An enlarged dataset with synthetic sequences appended.
    """
    num_materials, num_samples, _, _, _ = data.shape
    synthetic_data = []

    for material_idx in range(num_materials):
        material_data = data[material_idx]  # Shape: [samples, timesteps, taxels_x, taxels_y]
        material_synthetic = []

        for sample in material_data:
            # Reshape the sample for GMM input
            sample_flat = sample[-500:, :, :].reshape(-1)  # Use the last 500 timesteps as context
            
            # Generate synthetic data
            synthetic_flat = gmms[material_idx].sample(synthetic_length)[0]  # Shape: [synthetic_length * 16]
            synthetic = synthetic_flat.reshape(synthetic_length, 4, 4)  # Reshape back to [500, 4, 4]

            # Append the synthetic data to the original sample
            enlarged_sample = np.concatenate((sample, synthetic), axis=0)  # [1000+500, 4, 4]
            material_synthetic.append(enlarged_sample)

        synthetic_data.append(np.array(material_synthetic))  # Append all samples for this material

    return np.array(synthetic_data)


if __name__ == "__main__":
        
    # Load your original dataset
    with h5py.File("data/raw/tactmat.h5", "r") as dataset:
        samples = dataset["samples"][:]  # Shape: [materials, samples, timesteps, taxels_x, taxels_y]

    num_materials = samples.shape[0]

    # Train GMMs for each material
    gmms = train_gmms(samples, num_materials, n_components=5)

    # Generate synthetic samples
    enlarged_samples = generate_synthetic_samples(samples, gmms, sequence_length=1000, synthetic_length=500)

    # Save the augmented dataset
    with h5py.File("data/augmented_tactmat.h5", "w") as augmented_dataset:
        augmented_dataset.create_dataset("samples", data=enlarged_samples)
        augmented_dataset.create_dataset("materials", data=dataset["materials"][:])