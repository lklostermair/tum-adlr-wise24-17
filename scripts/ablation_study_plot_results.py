#!/usr/bin/env python
# plot_results.py

import os
import sys
import pickle
import matplotlib.pyplot as plt

def main():
    # Path to the results file created by train_multiple_models.py
    results_file = "output_multiple_seq/seq_length_results.pkl"

    with open(results_file, "rb") as f:
        results = pickle.load(f)

    # Example of each dict in results:
    # {
    #   "sequence_length": 50,
    #   "train_acc": 0.90,
    #   "val_acc": 0.85
    # }
    # Sort by sequence_length, just in case
    results = sorted(results, key=lambda d: d["sequence_length"])

    # Extract data
    seq_lengths = [r["sequence_length"] for r in results]
    train_accuracies = [r["train_acc"] for r in results]
    val_accuracies = [r["val_acc"] for r in results]

    # ------------------------
    # Plotting
    # ------------------------
    plt.figure(figsize=(8,6))



    # Plot validation accuracy
    plt.plot(seq_lengths, val_accuracies, label="Validation Accuracy", marker='o')

    plt.xlabel("Sequence Length")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join("output_multiple_seq", "accuracy_vs_seq_length.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to {fig_path}")

    plt.show()

if __name__ == "__main__":
    main()
