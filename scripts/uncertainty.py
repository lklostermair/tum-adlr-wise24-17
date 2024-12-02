import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import torch
import numpy as np
from utils.data_loader import TactileMaterialDataset
from models.TactNetII_model import TactNetII
from utils.monte_carlo import monte_carlo_inference

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path, input_channels=1, num_classes=36):
    """Load a trained model from the specified path."""
    model = TactNetII(input_channels=input_channels, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Evaluation mode
    return model

def evaluate_uncertainty(model, test_loader, num_samples=10):
    """Perform Monte Carlo Dropout to compute uncertainty."""
    uncertainties = []
    predictions = []
    ground_truths = []

    model.train()  # Enable dropout for MCD
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            mean_preds, uncertainty = monte_carlo_inference(model, X, num_samples=num_samples)
            uncertainties.append(uncertainty.cpu().numpy())
            predictions.append(torch.argmax(mean_preds, dim=1).cpu().numpy())
            ground_truths.append(y.cpu().numpy())

    return {
        "uncertainties": np.concatenate(uncertainties),
        "predictions": np.concatenate(predictions),
        "ground_truths": np.concatenate(ground_truths)
    }

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
    import pickle
    with open(class_uncertainty_save_path, "wb") as f:
        pickle.dump(class_uncertainties, f)
    print(f"Class-wise uncertainties saved at {class_uncertainty_save_path}")
