import sys
import os
import pickle
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data_loader import TactileMaterialDataset
from models.TactNetII_model import TactNetII
from scripts.train import train_model
from scripts.eval import evaluate_tactnet_mcdropout


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Directory to store overall results
    multi_output_dir = "output_multiple_seq"
    os.makedirs(multi_output_dir, exist_ok=True)

    # We'll run these sequence lengths
    seq_lengths = range(50, 1001, 50)

    # Common training hyperparameters
    num_epochs = 20       # For demonstration
    batch_size = 32
    learning_rate = 1e-4
    patience = 10

    results_list = []

    for seq_len in seq_lengths:
        print(f"\n--- Training model for sequence_length = {seq_len} ---")

        # Instantiate model
        model = TactNetII(input_channels=1, num_classes=36, sequence_length=seq_len).to(device)

        # Train
        train_info = train_model(
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            sequence_length=seq_len
        )

        # Extract final accuracies
        final_train_acc = train_info["final_train_acc"]
        final_val_acc = train_info["final_val_acc"]

        # Save to results
        results_list.append({
            "sequence_length": seq_len,
            "train_acc": final_train_acc,
            "val_acc": final_val_acc
        })

    # After training all models, save the results
    results_path = os.path.join(multi_output_dir, "seq_length_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results_list, f)

    print("\nAll models trained. Final results:")
    for r in results_list:
        print(f"Seq={r['sequence_length']}, TrainAcc={r['train_acc']*100:.2f}%, ValAcc={r['val_acc']*100:.2f}%")

    print(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()
