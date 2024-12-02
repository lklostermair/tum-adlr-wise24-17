import torch

def monte_carlo_inference(model, input_data, num_samples=100):
    """
    Perform Monte Carlo Dropout inference.

    Args:
        model (torch.nn.Module): Model with dropout layers.
        input_data (torch.Tensor): Input data.
        num_samples (int): Number of forward passes.

    Returns:
        tuple: (mean prediction, uncertainty)
            mean_prediction: Mean of predictions across Monte Carlo samples.
            uncertainty: Variance across Monte Carlo samples.
    """
    model.train()  # Keep dropout active
    predictions = []

    for _ in range(num_samples):
        predictions.append(model(input_data))  # Forward pass with dropout

    predictions = torch.stack(predictions)  # Shape: [num_samples, batch_size, num_classes]
    mean_prediction = predictions.mean(dim=0)  # Mean across MC samples
    uncertainty = predictions.var(dim=0)  # Variance across MC samples
    return mean_prediction, uncertainty
