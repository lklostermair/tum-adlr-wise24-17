import sys
import os
# Add project root to system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from utils.data_loader import TactileMaterialDataset
from models.TactNetII_model import TactNetII
from torch.utils.data import DataLoader

# Define output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
torch.cuda.empty_cache()

# Define function to calculate entropy
def calculate_entropy(probabilities):
    probabilities = np.clip(probabilities, 1e-10, 1.0)  # Clip probabilities to avoid log(0)
    return -np.sum(probabilities * np.log(probabilities), axis=-1)

# Load trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "output/best_model.pt"  # Update the path to your best model
model = TactNetII(input_channels=1, num_classes=36).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load dataset for evaluation
val_dataset = TactileMaterialDataset('data/raw/tactmat.h5', split='val', train_split=0.8)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
n_samples = 50  # Number of Monte Carlo samples

# Iterate over DataLoader and perform Monte Carlo Dropout Inference
all_predictions = []
all_variances = []
all_entropies = []
all_labels = []

for batch_idx, batch in enumerate(val_loader):
    print(f"Processing batch {batch_idx + 1}/{len(val_loader)}")
    input_data, labels = batch
    input_data = input_data.to(device)
    
    batch_predictions = []
    batch_size = input_data.size(0)
    minibatch_size = 4  # Further reduce mini-batch size to manage memory

    with torch.no_grad():
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_data = input_data[start:end]

            minibatch_preds = []
            for _ in range(n_samples):
                prediction = model(minibatch_data).cpu().detach()  # Move to CPU after forward pass
                minibatch_preds.append(prediction)

            minibatch_preds = torch.stack(minibatch_preds)  # Shape: [num_samples, minibatch_size, num_classes]
            batch_predictions.append(minibatch_preds)

            torch.cuda.empty_cache()  # Clear cache to free up memory

    # Concatenate minibatch predictions along the batch dimension
    batch_predictions = torch.cat(batch_predictions, dim=1)  # Shape: [num_samples, batch_size, num_classes]

    # Apply softmax to the model output
    mean_prediction = F.softmax(batch_predictions.mean(dim=0), dim=-1).numpy()

    variance = batch_predictions.var(dim=0).numpy()  # Variance across MC samples
    entropy = calculate_entropy(mean_prediction)  # Entropy of mean predictions

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

# Prepare entropy metrics dictionary
entropy_metrics = {
    'predictions': all_predictions,
    'variances': all_variances,
    'entropies': all_entropies,
    'labels': all_labels,
    'num_mc_samples': n_samples
}

# Save results as a comprehensive pickle file
entropy_metrics_path = os.path.join(output_dir, "entropy_metrics.pkl")
with open(entropy_metrics_path, 'wb') as f:
    pickle.dump(entropy_metrics, f)

print(f"Saved entropy metrics in {entropy_metrics_path}")