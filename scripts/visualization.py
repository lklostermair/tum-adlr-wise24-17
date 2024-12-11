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

# 1. Loss Curves Visualization (unchanged)
def plot_loss_curves():
    plt.figure(figsize=(10, 6))
    with open(os.path.join(output_dir, 'losses.pkl'), 'rb') as f:
        losses_data = pickle.load(f)  # Load the entire dictionary

    train_losses = losses_data["train_losses"]
    val_losses = losses_data["val_losses"]

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_loss_curves.png'))
    plt.close()

# 2. Confusion Matrix Visualization (MC)
def plot_confusion_matrix():
    with open(os.path.join(output_dir, 'evaluation_metrics_mc.pkl'), 'rb') as f:
        eval_metrics_mc = pickle.load(f)
    
    true_labels = eval_metrics_mc['true_labels']
    predicted_labels_mc = eval_metrics_mc['predicted_labels_mc']
    
    conf_matrix = confusion_matrix(true_labels, predicted_labels_mc)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=False, cmap='gray_r', linewidths=0.5, square=True, cbar=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_confusion_matrix.png'))
    plt.close()

# 3. Class-wise Accuracy Visualization (MC only)
def plot_class_wise_accuracy():
    # Load MC evaluation metrics
    with open(os.path.join(output_dir, 'evaluation_metrics_mc.pkl'), 'rb') as f:
        eval_metrics_mc = pickle.load(f)

    class_accuracies_mc = eval_metrics_mc['class_accuracies_mc']
    true_labels = eval_metrics_mc['true_labels']
    predicted_labels_mc = eval_metrics_mc['predicted_labels_mc']

    # Compute total MC accuracy
    overall_accuracy_mc = accuracy_score(true_labels, predicted_labels_mc) * 100

    # Sort by MC accuracy
    sorted_indices = np.argsort(class_accuracies_mc)[::-1]
    sorted_accuracies_mc = class_accuracies_mc[sorted_indices]

    plt.figure(figsize=(12, 5))
    x_positions = np.arange(len(sorted_accuracies_mc))

    plt.bar(x_positions, sorted_accuracies_mc, color='blue', label='Accuracy')

    plt.xticks(x_positions, sorted_indices, rotation=90)
    plt.xlabel('Material ID')
    plt.ylabel('Accuracy [%]')
    plt.axhline(y=np.mean(class_accuracies_mc), color='gray', linestyle='--', label='Mean Accuracy')
    plt.title(f'Per-Class MC Accuracy (Total: {overall_accuracy_mc:.2f}%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_per_class_accuracy.png'))
    plt.close()

# 4. Entropy Metrics Visualization (MC)
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
    
    mean_entropies = {cls: np.mean(entropies[cls]) for cls in entropies.keys()}
    sorted_classes_entropy = sorted(mean_entropies, key=mean_entropies.get)
    sorted_entropies = [entropies[cls] for cls in sorted_classes_entropy]
    
    sns.boxplot(data=sorted_entropies)
    plt.xticks(ticks=range(len(sorted_classes_entropy)), labels=sorted_classes_entropy)
    plt.ylabel('Entropy')
    plt.xlabel('Classes (Sorted by Mean Entropy)')
    plt.title('Class-wise Entropy Box Plot')
    
    for idx, cls in enumerate(sorted_classes_entropy):
        plt.plot([idx - 0.2, idx + 0.2], [mean_entropies[cls], mean_entropies[cls]], color='grey', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_class_wise_entropy.png'))
    plt.close()

def plot_confidence_vs_entropy():
    with open(os.path.join(output_dir, 'entropy_metrics.pkl'), 'rb') as f:
        entropy_metrics = pickle.load(f)
    
    all_entropies = entropy_metrics['entropies']
    all_labels = entropy_metrics['labels']
    all_predictions = entropy_metrics['predictions']
    
    confidences = np.max(all_predictions, axis=-1)
    entropy_values = all_entropies
    labels = all_labels
    
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    for cls in unique_labels:
        class_mask = (labels == cls)
        plt.scatter(confidences[class_mask], entropy_values[class_mask], alpha=0.6, label=f'Class {cls}')
    
    plt.xlabel('Confidence')
    plt.ylabel('Entropy')
    plt.title('Scatter Plot of Confidence vs. Entropy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_scatter_plot_confidence_vs_entropy_colored.png'))
    plt.close()

def plot_accuracy_vs_mean_entropy():
    with open(os.path.join(output_dir, 'evaluation_metrics_mc.pkl'), 'rb') as f:
        eval_metrics_mc = pickle.load(f)
    class_accuracies_mc = eval_metrics_mc['class_accuracies_mc']

    with open(os.path.join(output_dir, 'entropy_metrics.pkl'), 'rb') as f:
        entropy_metrics = pickle.load(f)
    
    all_entropies = entropy_metrics['entropies']
    all_labels = entropy_metrics['labels']
    
    unique_labels = np.unique(all_labels)
    mean_entropies = []
    std_entropies = []
    per_class_accuracies_mc = []
    
    for cls in unique_labels:
        class_entropies = all_entropies[all_labels == cls]
        mean_ent = np.mean(class_entropies)
        std_ent = np.std(class_entropies)
        acc_mc = class_accuracies_mc[cls]
        
        mean_entropies.append(mean_ent)
        std_entropies.append(std_ent)
        per_class_accuracies_mc.append(acc_mc)
    
    mean_entropies = np.array(mean_entropies)
    std_entropies = np.array(std_entropies)
    per_class_accuracies_mc = np.array(per_class_accuracies_mc)

    plt.figure(figsize=(10, 6))
    
    min_std, max_std = np.min(std_entropies), np.max(std_entropies)
    if max_std > min_std:
        normalized_std = (std_entropies - min_std) / (max_std - min_std)
    else:
        normalized_std = np.ones_like(std_entropies) * 0.5
    
    size_min, size_max = 50, 250
    sizes = size_min + normalized_std * (size_max - size_min)
    
    plt.scatter(per_class_accuracies_mc, mean_entropies, s=sizes, alpha=0.7)
    
    for i, cls in enumerate(unique_labels):
        plt.annotate(str(cls),
                     (per_class_accuracies_mc[i], mean_entropies[i]),
                     textcoords="offset points", xytext=(5,5), ha='left')
    
    coeffs = np.polyfit(per_class_accuracies_mc, mean_entropies, 1)
    linear_fit = np.poly1d(coeffs)
    x_fit = np.linspace(np.min(per_class_accuracies_mc), np.max(per_class_accuracies_mc), 100)
    y_fit = linear_fit(x_fit)
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label='Linear Fit')
    
    plt.xlabel('Per-Class MC Accuracy [%]')
    plt.ylabel('Mean Entropy')
    plt.title('Scatter Plot: Mean Entropy per Class vs. Accuracy (Point Size = Std of Entropy)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_scatter_mean_entropy_vs_accuracy.png'))
    plt.close()

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
