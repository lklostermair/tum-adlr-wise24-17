import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score

# Define output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

num_classes = 36  # or however many classes you have
cmap = plt.get_cmap('tab10', num_classes)
global_color_map = {cls_id: cmap(cls_id % 10) for cls_id in range(num_classes)}

# 1. Loss Curves Visualization
def plot_loss_curves():
    plt.figure(figsize=(10, 6))
    
    # Load training losses
    with open(os.path.join(output_dir, 'losses.pkl'), 'rb') as f:
            losses_data = pickle.load(f)  # Load the entire dictionary

    train_losses = losses_data["train_losses"]
    val_losses = losses_data["val_losses"]

    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('1. Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_loss_curves.png'))
    plt.close()

# 2. Confusion Matrix Visualization
def plot_confusion_matrix():
    with open(os.path.join(output_dir, 'evaluation_metrics.pkl'), 'rb') as f:
        eval_metrics = pickle.load(f)
    
    true_labels = eval_metrics['true_labels']
    predicted_labels = eval_metrics['predicted_labels']
    
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=False, cmap='gray_r', linewidths=0.5, square=True, cbar=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('2. Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_confusion_matrix.png'))
    plt.close()

# 3. Class-wise Accuracy Visualization
def plot_class_wise_accuracy():
    with open(os.path.join(output_dir, 'evaluation_metrics.pkl'), 'rb') as f:
        eval_metrics = pickle.load(f)
    
    true_labels = eval_metrics['true_labels']
    predicted_labels = eval_metrics['predicted_labels']
    class_accuracies = eval_metrics['class_accuracies']
    
    # Calculate total accuracy
    total_accuracy = accuracy_score(true_labels, predicted_labels) * 100
    
    # Sort accuracies
    sorted_indices = np.argsort(class_accuracies)[::-1]
    sorted_accuracies = np.array(class_accuracies)[sorted_indices]
    
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(sorted_accuracies)), sorted_accuracies, color='blue')
    plt.xticks(range(len(sorted_accuracies)), sorted_indices, rotation=90)
    plt.xlabel('Material ID')
    plt.ylabel('Accuracy [%]')
    plt.axhline(y=np.mean(class_accuracies), color='gray', linestyle='--', label='Mean Accuracy')
    plt.title(f'3. Per-Class Accuracy (Total Accuracy: {total_accuracy:.2f}%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_per_class_accuracy.png'))
    plt.close()

# 4. Entropy Metrics Visualization
def plot_entropy_metrics():
    with open(os.path.join(output_dir, 'entropy_metrics.pkl'), 'rb') as f:
        entropy_metrics = pickle.load(f)
    
    all_entropies = entropy_metrics['entropies']
    all_labels = entropy_metrics['labels']
    
    plt.figure(figsize=(12, 6))
    entropies = {}
    
    for cls in np.unique(all_labels):
        class_entropies = all_entropies[all_labels == cls]
        entropies[f'{cls}'] = class_entropies
    
    # Calculate mean entropy per class and sort
    mean_entropies = {cls: np.mean(entropies[cls]) for cls in entropies.keys()}
    sorted_classes_entropy = sorted(mean_entropies, key=mean_entropies.get)
    sorted_entropies = [entropies[cls] for cls in sorted_classes_entropy]
    
    sns.boxplot(data=sorted_entropies)
    plt.xticks(ticks=range(len(sorted_classes_entropy)), labels=sorted_classes_entropy)
    plt.ylabel('Entropy')
    plt.xlabel('Classes (Sorted by Mean Entropy)')
    plt.title('4. Class-wise Entropy Box Plot')
    
    # Plot mean entropy as grey dashed line
    for idx, cls in enumerate(sorted_classes_entropy):
        plt.plot([idx - 0.2, idx + 0.2], [mean_entropies[cls], mean_entropies[cls]], color='grey', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_class_wise_entropy.png'))
    plt.close()

def plot_confidence_vs_entropy():
    # Load data
    with open(os.path.join(output_dir, 'entropy_metrics.pkl'), 'rb') as f:
        entropy_metrics = pickle.load(f)
    
    all_entropies = entropy_metrics['entropies']
    all_labels = entropy_metrics['labels']
    all_predictions = entropy_metrics['predictions']
    
    # Compute confidence as the maximum predicted probability
    confidences = np.max(all_predictions, axis=-1)
    entropy_values = all_entropies
    labels = all_labels
    
    plt.figure(figsize=(10, 6))
    
    # Plot each class in the order of unique_labels so they match the default color cycle consistently
    unique_labels = np.unique(labels)
    for cls in unique_labels:
        class_mask = (labels == cls)
        # By not specifying any color, we rely on matplotlib's default color cycle
        plt.scatter(confidences[class_mask], entropy_values[class_mask], alpha=0.6, label=f'Class {cls}')
    
    plt.xlabel('Confidence')
    plt.ylabel('Entropy')
    plt.title('Scatter Plot of Confidence vs. Entropy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_plot_confidence_vs_entropy_colored.png'))
    plt.close()

def plot_accuracy_vs_mean_entropy():
    # Load evaluation metrics for class accuracies
    with open(os.path.join(output_dir, 'evaluation_metrics.pkl'), 'rb') as f:
        eval_metrics = pickle.load(f)
    class_accuracies = eval_metrics['class_accuracies']  # Assume array or list indexed by class ID

    # Load entropy metrics
    with open(os.path.join(output_dir, 'entropy_metrics.pkl'), 'rb') as f:
        entropy_metrics = pickle.load(f)
    
    all_entropies = entropy_metrics['entropies']
    all_labels = entropy_metrics['labels']
    
    # Compute mean and std of entropy per class
    unique_labels = np.unique(all_labels)
    mean_entropies = []
    std_entropies = []
    per_class_accuracies = []
    
    for cls in unique_labels:
        class_entropies = all_entropies[all_labels == cls]
        mean_ent = np.mean(class_entropies)
        std_ent = np.std(class_entropies)
        acc = class_accuracies[cls] if isinstance(class_accuracies, (list, np.ndarray)) else class_accuracies[str(cls)]
        
        mean_entropies.append(mean_ent)
        std_entropies.append(std_ent)
        per_class_accuracies.append(acc)
    
    mean_entropies = np.array(mean_entropies)
    std_entropies = np.array(std_entropies)
    per_class_accuracies = np.array(per_class_accuracies)

    plt.figure(figsize=(10, 6))
    
    # Normalize the std_entropies to a [0, 1] range
    min_std, max_std = np.min(std_entropies), np.max(std_entropies)
    if max_std > min_std:
        normalized_std = (std_entropies - min_std) / (max_std - min_std)
    else:
        # If all std_entropies are the same, just use a uniform size
        normalized_std = np.ones_like(std_entropies) * 0.5
    
    # Scale normalized sizes to a desired range, e.g. [50, 250]
    size_min, size_max = 50, 250
    sizes = size_min + normalized_std * (size_max - size_min)
    
    # Plot points (single color) with normalized sizes
    plt.scatter(per_class_accuracies, mean_entropies, s=sizes, alpha=0.7)
    
    # Annotate each point with its class label
    for i, cls in enumerate(unique_labels):
        plt.annotate(str(cls),
                     (per_class_accuracies[i], mean_entropies[i]),
                     textcoords="offset points", xytext=(5,5), ha='left')
    
    # Compute and plot a linear approximation line
    coeffs = np.polyfit(per_class_accuracies, mean_entropies, 1)
    linear_fit = np.poly1d(coeffs)
    x_fit = np.linspace(np.min(per_class_accuracies), np.max(per_class_accuracies), 100)
    y_fit = linear_fit(x_fit)
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label='Linear Fit')
    
    plt.xlabel('Per-Class Accuracy [%]')
    plt.ylabel('Mean Entropy')
    plt.title('Scatter Plot: Mean Entropy per Class vs. Accuracy (Point Size = Std of Entropy)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_mean_entropy_vs_accuracy.png'))
    plt.close()




# Run all visualizations
def main():
    plot_loss_curves()
    plot_confusion_matrix()
    plot_class_wise_accuracy()
    plot_entropy_metrics()
    plot_confidence_vs_entropy()
    plot_accuracy_vs_mean_entropy()

    print("All visualizations have been saved in the output directory.")

if __name__ == "__main__":
    main()