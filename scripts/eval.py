import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from utils.data_loader import TactileMaterialDataset
from models.TactNetII_model import TactNetII
from torch.utils.data import DataLoader
import pickle

# Load trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "output/best_model.pt"  # Update the path to your best model
model = TactNetII(input_channels=1, num_classes=36).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load dataset for evaluation
val_dataset = TactileMaterialDataset('data/raw/tactmat.h5', split='val', train_split=0.8)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Evaluate model and collect predictions
true_labels = []
predicted_labels = []

with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
        
        true_labels.extend(y.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

# Convert to NumPy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=False, cmap='gray_r', linewidths=0.5, square=True, cbar=True)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("output/confusion_matrix.png")
plt.show()

# Calculate per-class accuracy
num_classes = len(np.unique(true_labels))
class_accuracies = []

for class_id in range(num_classes):
    class_indices = (true_labels == class_id)
    class_acc = accuracy_score(true_labels[class_indices], predicted_labels[class_indices])
    class_accuracies.append(class_acc * 100)

# Sort by accuracy for better visualization
sorted_indices = np.argsort(-np.array(class_accuracies))
sorted_accuracies = np.array(class_accuracies)[sorted_indices]

# Plot Accuracy per Class
plt.figure(figsize=(12, 5))
plt.bar(range(num_classes), sorted_accuracies, color='blue')
plt.xticks(range(num_classes), sorted_indices, rotation=90)
plt.xlabel('material ID')
plt.ylabel('accuracy [%]')
plt.axhline(y=np.mean(class_accuracies), color='gray', linestyle='--')
plt.title('Per Class Accuracy')
plt.tight_layout()
plt.savefig("output/per_class_accuracy.png")
plt.show()

# Load loss data
output_dir = "output"  # Directory where the loss data is saved
losses_path = f"{output_dir}/losses.pkl"

with open(losses_path, "rb") as f:
    losses_data = pickle.load(f)

train_losses = losses_data["train_losses"]
val_losses = losses_data["val_losses"]

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
epochs = range(1, len(train_losses) + 1)

plt.plot(epochs, train_losses, label='Training Loss', color='blue')
plt.plot(epochs, val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/training_val_loss.png")
plt.show()
