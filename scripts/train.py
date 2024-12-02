import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data_loader import TactileMaterialDataset
from models.TactNetII_model import TactNetII
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_dir = "output"  # Define the output directory
os.makedirs(output_dir, exist_ok=True)

def train_model(model, num_epochs=100, batch_size=32, learning_rate=1e-4, patience=10):
    print("Initializing dataset...")
    # Initialize dataset
    train_dataset = TactileMaterialDataset('data/raw/tactmat.h5', split='train', train_split=0.8)
    val_dataset = TactileMaterialDataset('data/raw/tactmat.h5', split='val', train_split=0.8)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=170, gamma=0.1)

    # Tracking metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    no_improve_count = 0

    print("Starting training...")
    # Training loop
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation Phase
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                # Forward pass
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                # Accuracy calculation
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == y).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"Saved best model at epoch {epoch + 1}")
            no_improve_count = 0
        else:
            no_improve_count += 1

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy * 100:.2f}%")
        
        if no_improve_count >= patience:
            print("Early stopping triggered.")
            break

        # Step the scheduler
        scheduler.step()

    # Save metrics
    with open(os.path.join(output_dir, "losses.pkl"), "wb") as f:
        pickle.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    with open(os.path.join(output_dir, "val_accuracies.pkl"), "wb") as f:
        pickle.dump({"val_accuracies": val_accuracies}, f)

    print("Training complete.")
    return {
        "best_model_path": os.path.join(output_dir, "best_model.pt"),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }

if __name__ == "__main__":

    # Instantiate model

    model = TactNetII(input_channels=1, num_classes=36).to(device)
    
    # Train the model
    train_model(
        model=model,
        num_epochs=100,
        batch_size=32,
        patience=10
    )
