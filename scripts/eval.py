import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from utils.data_loader import TactileMaterialDataset
from models.TactNetII_model import TactNetII
from torch.utils.data import DataLoader
import pickle
import scipy.stats as stats

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

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
predicted_probs = []

with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        true_labels.extend(y.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())
        predicted_probs.extend(probs.cpu().numpy())

# Convert to NumPy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
predicted_probs = np.array(predicted_probs)

# Compute evaluation metrics
num_classes = len(np.unique(true_labels))

# Calculate per-class accuracy
class_accuracies = []
for class_id in range(num_classes):
    class_indices = (true_labels == class_id)
    class_acc = accuracy_score(true_labels[class_indices], predicted_labels[class_indices])
    class_accuracies.append(class_acc * 100)

# Sort accuracies
sorted_indices = np.argsort(-np.array(class_accuracies))
sorted_accuracies = np.array(class_accuracies)[sorted_indices]

# Prepare comprehensive metrics dictionary
evaluation_metrics = {
    'true_labels': true_labels,
    'predicted_labels': predicted_labels,
    'predicted_probs': predicted_probs,
    'class_accuracies': class_accuracies,
    'sorted_class_indices': sorted_indices,
    'sorted_class_accuracies': sorted_accuracies,
    'overall_accuracy': accuracy_score(true_labels, predicted_labels),
    'confusion_matrix': confusion_matrix(true_labels, predicted_labels)
}


# Additional detailed classification report
classification_rep = classification_report(true_labels, predicted_labels, output_dict=True)
evaluation_metrics['classification_report'] = classification_rep

# Save metrics to a pickle file
metrics_path = os.path.join(output_dir, 'evaluation_metrics.pkl')
with open(metrics_path, 'wb') as f:
    pickle.dump(evaluation_metrics, f)

print(f"Saved evaluation metrics to {metrics_path}")
print(f"Overall Accuracy: {evaluation_metrics['overall_accuracy']* 100:.2f}%")