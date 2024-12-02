import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import torch

def plot_loss_curve(train_losses, val_losses, save_path=None):
    """
    Plot the training and validation loss curve.

    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curve saved at {save_path}")
    else:
        plt.show()

def plot_accuracy_curve(val_accuracies, save_path=None):
    """
    Plot the validation accuracy curve.
    
    """
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        print(f"Accuracy curve saved at {save_path}")
    else:
        plt.show()

def plot_confusion_matrix(labels, preds, class_names, save_path=None):
    """
    Plot the confusion matrix.
    
    """
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="viridis", xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved at {save_path}")
    else:
        plt.show()

def plot_uncertainty(uncertainties, save_path=None):
    """
    Plot the uncertainty distribution.
    
    """
    plt.figure(figsize=(10, 5))
    plt.hist(uncertainties.mean(axis=1), bins=20, alpha=0.7, color='blue')
    plt.xlabel("Uncertainty")
    plt.ylabel("Frequency")
    plt.title("Uncertainty Distribution")
    if save_path:
        plt.savefig(save_path)
        print(f"Uncertainty plot saved at {save_path}")
    else:
        plt.show()

def plot_class_uncertainty(class_uncertainties, save_path=None):
    """Visualize class-wise uncertainty."""
    classes = list(class_uncertainties.keys())
    uncertainties = list(class_uncertainties.values())

    plt.figure(figsize=(12, 6))
    plt.bar(classes, uncertainties, color="blue", alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel("Average Uncertainty")
    plt.title("Class-wise Average Uncertainty")
    plt.xticks(classes)
    if save_path:
        plt.savefig(save_path)
        print(f"Class-wise uncertainty plot saved at {save_path}")
    else:
        plt.show()
        
if __name__ == "__main__":
    

    # Define paths
    output_dir = "output"
    losses_path = os.path.join(output_dir, "losses.pkl")
    val_accuracies_path = os.path.join(output_dir, "val_accuracies.pkl")
    uncertainty_results_path = os.path.join(output_dir, "uncertainty_results.pt")
    confusion_matrix_save_path = os.path.join(output_dir, "confusion_matrix.png")
    loss_curve_save_path = os.path.join(output_dir, "loss_curve.png")
    accuracy_curve_save_path = os.path.join(output_dir, "accuracy_curve.png")
    uncertainty_plot_save_path = os.path.join(output_dir, "uncertainty_distribution.png")
    class_uncertainty_path = "output/class_uncertainties.pkl"
    class_uncertainty_plot_path = "output/class_uncertainty.png"

    # Load losses
    if os.path.exists(losses_path):
        with open(losses_path, "rb") as f:
            losses = pickle.load(f)
            train_losses = losses["train_losses"]
            val_losses = losses["val_losses"]
        print("Loaded loss metrics.")

        # Plot loss curve
        plot_loss_curve(train_losses, val_losses, save_path=loss_curve_save_path)

    # Load validation accuracies
    if os.path.exists(val_accuracies_path):
        with open(val_accuracies_path, "rb") as f:
            val_accuracies_data = pickle.load(f)
            val_accuracies = val_accuracies_data["val_accuracies"]
        print("Loaded validation accuracies.")

        # Plot accuracy curve
        plot_accuracy_curve(val_accuracies, save_path=accuracy_curve_save_path)

    # Load uncertainty results
    if os.path.exists(uncertainty_results_path):
        uncertainty_results = torch.load(uncertainty_results_path)
        uncertainties = uncertainty_results["uncertainties"]
        ground_truths = uncertainty_results["ground_truths"]
        predictions = uncertainty_results["predictions"]
        print("Loaded uncertainty results.")

        # Plot uncertainty distribution
        plot_uncertainty(uncertainties, save_path=uncertainty_plot_save_path)

        # Plot confusion matrix
        plot_confusion_matrix(
            labels=ground_truths,
            preds=predictions,
            class_names=[f"Class {i}" for i in range(len(set(ground_truths)))],
            save_path=confusion_matrix_save_path,
        )

    # Load class uncertainty results    
    if os.path.exists(class_uncertainty_path):
        with open(class_uncertainty_path, "rb") as f:
            class_uncertainties = pickle.load(f)
        print("Loaded class-wise uncertainties.")

        # Visualize
        plot_class_uncertainty(class_uncertainties, save_path=class_uncertainty_plot_path)
    else:
        print(f"Class uncertainty file not found at {class_uncertainty_path}")


    print("Visualization complete. Check the output directory for saved plots.")
