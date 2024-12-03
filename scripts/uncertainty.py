import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.monte_carlo import monte_carlo_inference
from utils.data_loader import TactileMaterialDataset
from models.TactNetII_model import TactNetII
from torch.utils.data import DataLoader
import torch

# Add project root to system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Define output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Define function to calculate entropy
def calculate_entropy(probabilities):
    return -np.sum(probabilities * np.log(probabilities + 1e-10), axis=-1)

# Load trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "output/best_model.pt"  # Update the path to your best model
model = TactNetII(input_channels=1, num_classes=36).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load dataset for evaluation
val_dataset = TactileMaterialDataset('data/raw/tactmat.h5', split='val', train_split=0.8)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
n_samples = 100  # Number of Monte Carlo samples

# Iterate over DataLoader and perform Monte Carlo Dropout Inference
all_predictions = []
all_variances = []
all_entropies = []
all_labels = []

for batch in val_loader:
    input_data, labels = batch
    input_data = input_data.to(device)

    # Perform Monte Carlo Dropout Inference for the current batch
    result = monte_carlo_inference(model, input_data, n_samples=n_samples)
    mean_prediction = result["mean_prediction"].cpu().detach().numpy()
    variance = result["variance"].cpu().detach().numpy()
    entropy = result["entropy"].cpu().detach().numpy()

    # Store the predictions, variances, entropies, and labels
    all_predictions.append(mean_prediction)
    all_variances.append(variance)
    all_entropies.append(entropy)
    all_labels.append(labels.numpy())

# Convert lists to numpy arrays for further processing
all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: [total_samples, num_classes]
all_variances = np.concatenate(all_variances, axis=0)  # Shape: [total_samples, num_classes]
all_entropies = np.concatenate(all_entropies, axis=0)  # Shape: [total_samples]
all_labels = np.concatenate(all_labels, axis=0)  # Shape: [total_samples]

# Calculate Entropy and Variance per Class
entropies = {}
variances = {}
classes = all_predictions.shape[-1]

for cls in range(classes):
    # Extract entropies and variances for the current class
    class_entropies = all_entropies[all_labels == cls]
    class_variances = all_variances[all_labels == cls, cls]
    entropies[f'class_{cls}'] = class_entropies
    variances[f'class_{cls}'] = class_variances

# Plotting

# 1. Class-Wise Uncertainty Box Plot (Entropy)
mean_entropies = {cls: np.mean(entropies[cls]) for cls in entropies.keys()}
sorted_classes_entropy = sorted(mean_entropies, key=mean_entropies.get)
sorted_entropies = [entropies[cls] for cls in sorted_classes_entropy]

plt.figure(figsize=(12, 6))
sns.boxplot(data=sorted_entropies)
plt.xticks(ticks=range(len(sorted_classes_entropy)), labels=sorted_classes_entropy)
plt.ylabel('Entropy (Uncertainty)')
plt.xlabel('Classes (Sorted by Mean Entropy)')
plt.title('Class-wise Uncertainty (Entropy) Box Plot')

# Plot mean uncertainty as grey dashed line
for idx, cls in enumerate(sorted_classes_entropy):
    plt.plot([idx - 0.2, idx + 0.2], [mean_entropies[cls], mean_entropies[cls]], color='grey', linestyle='--')

plt.savefig(os.path.join(output_dir, 'class_wise_uncertainty_boxplot_entropy.png'))
plt.close()

# 2. Scatter Plot of Confidence vs. Uncertainty (Entropy)
confidences = np.max(all_predictions, axis=-1)  # Confidence as the maximum mean class probability
entropy_values = all_entropies

plt.figure(figsize=(10, 6))
plt.scatter(confidences, entropy_values, alpha=0.6)
plt.xlabel('Confidence')
plt.ylabel('Entropy (Uncertainty)')
plt.title('Scatter Plot of Confidence vs. Uncertainty')
plt.savefig(os.path.join(output_dir, 'scatter_plot_confidence_vs_uncertainty.png'))
plt.close()

# 3. Variance Box Plot with Whiskers for Each Class (Sorted)
mean_variances = {cls: np.mean(variances[cls]) for cls in variances.keys()}
sorted_classes_variance = sorted(mean_variances, key=mean_variances.get)
sorted_variances = [variances[cls] for cls in sorted_classes_variance]

plt.figure(figsize=(12, 6))
sns.boxplot(data=sorted_variances)
plt.xticks(ticks=range(len(sorted_classes_variance)), labels=sorted_classes_variance)
plt.ylabel('Variance')
plt.xlabel('Classes (Sorted by Mean Variance)')
plt.title('Class-wise Variance Box Plot with Whiskers (Sorted)')
plt.savefig(os.path.join(output_dir, 'class_wise_variance_boxplot.png'))
plt.close()
