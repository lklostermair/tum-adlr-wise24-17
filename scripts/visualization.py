import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
from utils.data_loader import TactileMaterialDataset
from models.TactNetII_model import TactNetII
from utils.monte_carlo import monte_carlo_inference

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
def load_model(model_path, input_channels=1, num_classes=36):
    """Load a trained model from the specified path."""
    model = TactNetII(input_channels=input_channels, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Evaluation mode
    return model

# Evaluate uncertainty
def evaluate_uncertainty(model, test_loader, num_samples=10):
    """Perform Monte Carlo Dropout to compute uncertainty and evaluate accuracy."""
    uncertainties = []
    predictions = []
    ground_truths = []

    # Use train mode for MCD
    model.train()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            mean_preds, uncertainty = monte_carlo_inference(model, X, num_samples=num_samples)
            uncertainties.append(uncertainty.cpu().numpy())
            predictions.append(torch.argmax(mean_preds, dim=1).cpu().numpy())
            ground_truths.append(y.cpu().numpy())

    # Also evaluate without dropout for comparison
    model.eval()
    eval_predictions = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = model(X)
            eval_predictions.append(torch.argmax(preds, dim=1).cpu().numpy())

    eval_predictions = np.concatenate(eval_predictions)
    overall_accuracy = np.mean(eval_predictions == np.concatenate(ground_truths))
    print(f"Accuracy in evaluation mode: {overall_accuracy * 100:.2f}%")

    return {
        "uncertainties": np.concatenate(uncertainties),
        "predictions": np.concatenate(predictions),
        "ground_truths": np.concatenate(ground_truths),
        "eval_accuracy": overall_accuracy
    }

# Compute class uncertainty
def compute_class_uncertainty(uncertainties, ground_truths, num_classes=36):
    """Compute average uncertainty for each class."""
    class_uncertainties = {cls: [] for cls in range(num_classes)}

    for uncertainty, label in zip(uncertainties, ground_truths):
        class_uncertainties[label].append(uncertainty.mean())  # Mean uncertainty for the sample

    # Compute the average uncertainty per class
    avg_class_uncertainties = {
        cls: np.mean(class_uncertainties[cls]) if class_uncertainties[cls] else 0.0
        for cls in class_uncertainties
    }
    return avg_class_uncertainties

# Main block
if __name__ == "__main__":
    # Paths
    model_path = "output/best_model.pt"
    test_data_path = "data/raw/tactmat.h5"
    uncertainty_results_save_path = "output/uncertainty_results.pt"
    class_uncertainty_save_path = "output/class_uncertainties.pkl"

    # Load model
    print("Loading model...")
    model = load_model(model_path)

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = TactileMaterialDataset(test_data_path, split='val', train_split=0.8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate uncertainty
    print("Evaluating uncertainty...")
    results = evaluate_uncertainty(model, test_loader, num_samples=10)

    # Save uncertainty results
    os.makedirs("output", exist_ok=True)
    torch.save(results, uncertainty_results_save_path)
    print(f"Uncertainty results saved at {uncertainty_results_save_path}")

    # Compute class-wise uncertainty
    print("Computing class-wise uncertainty...")
    class_uncertainties = compute_class_uncertainty(
        results["uncertainties"], results["ground_truths"], num_classes=36
    )

    # Save class-wise uncertainties
    with open(class_uncertainty_save_path, "wb") as f:
        pickle.dump(class_uncertainties, f)
    print(f"Class-wise uncertainties saved at {class_uncertainty_save_path}")

    # Loss and Validation Accuracy Curve
    with open(os.path.join("output", 'losses.pkl'), 'rb') as f:
        losses_data = pickle.load(f)
        train_losses = losses_data['train_losses']
        val_losses = losses_data['val_losses']

    with open(os.path.join("output", 'val_accuracies.pkl'), 'rb') as f:
        val_accuracies = pickle.load(f)['val_accuracies']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(range(1, len(train_losses) + 1), train_losses, 'g-', label='Train Loss')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'b-', label='Validation Accuracy')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='g')
    ax2.set_ylabel('Validation Accuracy', color='b')
    plt.title('Loss and Validation Accuracy Curve')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig(os.path.join("output", 'loss_val_accuracy_curve.png'))
    plt.close()

    # Confusion Matrix Heatmap
    true_labels = results['ground_truths']
    pred_labels = results['predictions']
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Greys', xticklabels=[f'Material {i}' for i in range(36)], yticklabels=[f'Material {i}' for i in range(36)], cbar=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join("output", 'confusion_matrix.png'))
    plt.close()

    # Class-wise Uncertainty Bar Plot
    class_mean_uncertainties = [class_uncertainties[c] for c in range(36)]
    sorted_indices = np.argsort(class_mean_uncertainties)[::-1]

    plt.bar(np.array([f'Material {i}' for i in range(36)])[sorted_indices], np.array(class_mean_uncertainties)[sorted_indices])
    plt.axhline(np.mean(class_mean_uncertainties), color='r', linestyle='--', label='Average Uncertainty')
    plt.xlabel('Material IDs')
    plt.ylabel('Uncertainty')
    plt.xticks(rotation=90)
    plt.title('Class-wise Uncertainty')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("output", 'class_wise_uncertainty.png'))
    plt.close()

    # Class-wise Accuracy Bar Plot
    class_accuracies = [np.sum((np.array(results['ground_truths']) == c) & (np.array(results['predictions']) == c)) / np.sum(np.array(results['ground_truths']) == c) if np.sum(np.array(results['ground_truths']) == c) > 0 else 0 for c in range(36)]  # Compute class accuracies manually

    sorted_indices_acc = np.argsort(class_accuracies)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(np.array([f'Material {i}' for i in range(36)])[sorted_indices_acc], np.array(class_accuracies)[sorted_indices_acc], color='skyblue')
    plt.axhline(np.mean(class_accuracies), color='r', linestyle='--', label='Average Accuracy')
    plt.xlabel('Material IDs')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.title('Class-wise Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("output", 'class_wise_accuracy.png'))
    plt.close()