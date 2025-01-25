import os
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

# Import your model and dataset classes
# Make sure these imports point to the correct locations in your project
# from models.TactNetII_model import TactNetII
# from utils.data_loader import TactileMaterialDataset

def calculate_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Utility function to calculate Shannon entropy from a probability distribution."""
    probabilities = np.clip(probabilities, 1e-10, 1.0)  # Avoid log(0)
    return -np.sum(probabilities * np.log(probabilities), axis=-1)

def evaluate_tactnet_mcdropout(
    model_path: str,
    data_file: str,
    TactNetII_class,
    TactileMaterialDataset_class,
    output_dir: str = "output",
    input_channels: int = 1,
    num_classes: int = 36,
    device: str = None,
    batch_size: int = 32,
    n_samples: int = 100,
    minibatch_size: int = 4,
    train_split: float = 0.8
):
    """
    Evaluate a trained TactNetII model with Monte Carlo Dropout on a validation split.

    Args:
        model_path (str): Path to the trained model file (e.g., 'output/best_model.pt').
        data_file (str): Path to the HDF5 file containing your tactile dataset.
        TactNetII_class: The class definition for your TactNetII model.
        TactileMaterialDataset_class: The class definition for your custom dataset loader.
        output_dir (str): Directory where evaluation metrics (pickles) will be saved.
        input_channels (int): Number of input channels for TactNetII (default=1).
        num_classes (int): Number of classes/materials (default=36).
        device (str): Device to use ('cuda:0' or 'cpu'). If None, auto-detects GPU if available.
        batch_size (int): Batch size for the DataLoader (default=32).
        n_samples (int): Number of Monte Carlo samples for dropout inference (default=100).
        minibatch_size (int): Sub-splitting each batch into smaller chunks to reduce memory usage (default=4).
        train_split (float): Proportion of data to use for training, rest is used for validation (default=0.8).

    Returns:
        dict: A dictionary containing evaluation metrics:
              - "entropy_metrics": A dictionary with 'predictions', 'variances', 'entropies', 'labels', 'num_mc_samples', 'accuracy'
              - "evaluation_metrics_mc": A dictionary with 'true_labels', 'predicted_labels_mc', 'predicted_probs_mc',
                'class_accuracies_mc', 'sorted_class_indices_mc', 'sorted_class_accuracies_mc',
                'overall_accuracy_mc', 'confusion_matrix_mc'
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect device if not provided
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Clear CUDA cache (optional, if memory is tight)
    torch.cuda.empty_cache()

    # 1. Load the pre-trained model
    model = TactNetII_class(input_channels=input_channels, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load validation dataset and create DataLoader
    val_dataset = TactileMaterialDataset_class(file_path=data_file, split='val', train_split=train_split)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Prepare to accumulate results
    all_predictions = []
    all_variances = []
    all_entropies = []
    all_labels = []

    # 4. Loop over validation batches
    for batch_idx, (input_data, labels) in enumerate(val_loader):
        print(f"Processing batch {batch_idx + 1}/{len(val_loader)}")

        # Move inputs to the device
        input_data = input_data.to(device)

        # We'll store predictions for MC sampling
        batch_predictions = []
        batch_size_current = input_data.size(0)

        with torch.no_grad():
            # 4a. Break the current batch into smaller mini-batches to save memory
            for start in range(0, batch_size_current, minibatch_size):
                end = start + minibatch_size
                minibatch_data = input_data[start:end]
                minibatch_preds = []

                # 4b. Monte Carlo sampling
                for _ in range(n_samples):
                    prediction = model(minibatch_data).cpu().detach()
                    minibatch_preds.append(prediction)

                minibatch_preds = torch.stack(minibatch_preds)  # Shape: [n_samples, mini_batch, num_classes]
                batch_predictions.append(minibatch_preds)

                # Clear CUDA cache
                torch.cuda.empty_cache()

        # 4c. Concatenate minibatch predictions along the batch dimension
        # Shape after cat: [n_samples, batch_size_current, num_classes]
        batch_predictions = torch.cat(batch_predictions, dim=1)

        # 4d. Compute mean predictions (MC mean) and apply softmax
        mean_prediction = F.softmax(batch_predictions.mean(dim=0), dim=-1).numpy()  # [batch_size_current, num_classes]

        # 4e. Compute variance across MC samples
        variance = batch_predictions.var(dim=0).numpy()  # [batch_size_current, num_classes]

        # 4f. Compute entropy of the mean predictions
        entropy = calculate_entropy(mean_prediction)

        # 4g. Accumulate results
        all_predictions.append(mean_prediction)
        all_variances.append(variance)
        all_entropies.append(entropy)
        all_labels.append(labels.numpy())

    # 5. Convert lists to NumPy arrays
    all_predictions = np.concatenate(all_predictions, axis=0) 
    all_variances = np.concatenate(all_variances, axis=0)
    all_entropies = np.concatenate(all_entropies, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)           

    # 6. Compute accuracy
    predicted_labels = np.argmax(all_predictions, axis=1)
    accuracy = np.mean(predicted_labels == all_labels)
    print(f"MC Accuracy: {accuracy * 100:.2f}%")

    ################################
    # Prepare and Save Entropy Data
    ################################
    entropy_metrics = {
        'predictions': all_predictions,
        'variances': all_variances,
        'entropies': all_entropies,
        'labels': all_labels,
        'num_mc_samples': n_samples,
        'accuracy': accuracy
    }

    entropy_metrics_path = os.path.join(output_dir, "entropy_metrics.pkl")
    with open(entropy_metrics_path, 'wb') as f:
        pickle.dump(entropy_metrics, f)
    print(f"Saved entropy metrics in {entropy_metrics_path}")

    #########################################
    # Compute Class-Wise Accuracies with MC
    #########################################
    num_classes_unique = len(np.unique(all_labels))
    class_accuracies_mc = np.zeros(num_classes_unique)
    for cls in range(num_classes_unique):
        cls_mask = (all_labels == cls)
        if np.sum(cls_mask) > 0:
            class_accuracies_mc[cls] = np.mean(predicted_labels[cls_mask] == all_labels[cls_mask]) * 100.0
        else:
            class_accuracies_mc[cls] = 0.0

    overall_accuracy_mc = accuracy_score(all_labels, predicted_labels)
    sorted_indices_mc = np.argsort(class_accuracies_mc)[::-1]
    sorted_class_accuracies_mc = class_accuracies_mc[sorted_indices_mc]
    confusion_matrix_mc = confusion_matrix(all_labels, predicted_labels)

    evaluation_metrics_mc = {
        'true_labels': all_labels,
        'predicted_labels_mc': predicted_labels,
        'predicted_probs_mc': all_predictions,  # MC-averaged predictions
        'class_accuracies_mc': class_accuracies_mc,
        'sorted_class_indices_mc': sorted_indices_mc,
        'sorted_class_accuracies_mc': sorted_class_accuracies_mc,
        'overall_accuracy_mc': overall_accuracy_mc,
        'confusion_matrix_mc': confusion_matrix_mc
    }

    evaluation_metrics_mc_path = os.path.join(output_dir, "evaluation_metrics_mc.pkl")
    with open(evaluation_metrics_mc_path, 'wb') as f:
        pickle.dump(evaluation_metrics_mc, f)
    print(f"MC evaluation metrics saved in {evaluation_metrics_mc_path}")

    # Return the metrics as a dictionary for direct use in Python
    return {
        "entropy_metrics": entropy_metrics,
        "evaluation_metrics_mc": evaluation_metrics_mc
    }
