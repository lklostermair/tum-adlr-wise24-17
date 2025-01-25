import sys
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data_loader import TactileMaterialDataset
from models.TactNetII_model import TactNetII

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_dir = "output"  
os.makedirs(output_dir, exist_ok=True)

def train_model(
    model,
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    patience=10,
    sequence_length=1000,
):
    """
    Trains the given TactNetII model on a dataset (train + val) created
    for a specified sequence_length, with early stopping.

    Returns a dictionary containing metrics, including best accuracies and paths.
    """

    print("Initializing dataset...")
    # Initialize dataset
    train_dataset = TactileMaterialDataset(
        file_path='data/raw/tactmat.h5',
        split='train',
        train_split=0.8,
        sequence_length=sequence_length,
        data_augment=True
    )
    val_dataset = TactileMaterialDataset(
        file_path='data/raw/tactmat.h5',
        split='val',
        train_split=0.8,
        sequence_length=sequence_length,
        data_augment=False
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    
    # Loss, optimizer, and LR scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=170, gamma=0.1)

    # Tracking metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_val_acc = 0.0       # Will store the best validation accuracy
    best_train_acc = 0.0     # Training accuracy at the time we got best val
    best_epoch = 0
    no_improve_count = 0

    best_model_path = os.path.join(output_dir, f"best_model_seq_{sequence_length}.pt")

    losses_filename = f"losses_seq_{sequence_length}.pkl"
    accuracies_filename = f"val_accuracies_seq_{sequence_length}.pkl"

    print("Starting training...")
    for epoch in range(num_epochs):
        # -----------------------------
        # 1) Training Phase
        # -----------------------------
        model.train()
        train_loss = 0.0
        train_correct = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Compute training accuracy for this mini-batch
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == y).sum().item()

        # Average training loss, accuracy
        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_dataset)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # -----------------------------
        # 2) Validation Phase
        # -----------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == y).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # -----------------------------
        # 3) Check if this is a new best
        # -----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_accuracy
            best_train_acc = train_acc   # Training accuracy at this best epoch
            best_epoch = epoch + 1

            # Save model checkpoint
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {best_epoch}: {best_model_path}")
            no_improve_count = 0
        else:
            no_improve_count += 1

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc*100:.2f}%, "
            f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy*100:.2f}%"
        )

        if no_improve_count >= patience:
            print("Early stopping triggered.")
            break

        # Step LR scheduler
        scheduler.step()

    # -----------------------------
    # 4) Final Saves
    # -----------------------------
    # Save the loss curves
    with open(os.path.join(output_dir, losses_filename), "wb") as f:
        pickle.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    # Save the accuracy curves
    with open(os.path.join(output_dir, accuracies_filename), "wb") as f:
        pickle.dump({
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }, f)

    print("Training complete.")
    print(f"Best epoch = {best_epoch}, Best Val Acc = {best_val_acc*100:.2f}%")

    # Return best accuracies, not the final ones
    return {
        "best_model_path": best_model_path,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_train_acc": best_train_acc,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }

if __name__ == "__main__":
    seq_len = 1000
    model = TactNetII(input_channels=1, num_classes=36, sequence_length=seq_len).to(device)

    results = train_model(
        model=model,
        num_epochs=100,
        batch_size=32,
        patience=15,
        sequence_length=seq_len
    )
    print("Best epoch:", results["best_epoch"])
    print("Best val accuracy: {:.2f}%".format(results["best_val_acc"] * 100))
    print("Corresponding train accuracy: {:.2f}%".format(results["best_train_acc"] * 100))
