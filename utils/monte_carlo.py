import torch

def monte_carlo_inference(model, input_data, num_samples=100, return_raw=False):
    """
    Perform Monte Carlo Dropout inference.
    Args:
        model (torch.nn.Module): Model with dropout layers.
        input_data (torch.Tensor): Input data.
        num_samples (int): Number of forward passes.
        return_raw (bool): If True, return raw predictions.
    Returns:
        dict: Contains mean prediction, variance, and optionally raw predictions.
    """
    model.train()  # Keep dropout active
    predictions = []

    for _ in range(num_samples):
        predictions.append(model(input_data))  # Forward pass with dropout

    predictions = torch.stack(predictions)  # Shape: [num_samples, batch_size, num_classes]
    mean_prediction = predictions.mean(dim=0)  # Mean across MC samples
    variance = predictions.var(dim=0)  # Variance across MC samples
    entropy = -torch.sum(mean_prediction * torch.log(mean_prediction + 1e-10), dim=-1)  # Entropy of mean predictions

    result = {
        "mean_prediction": mean_prediction,
        "variance": variance,
        "entropy": entropy,
    }
    
    if return_raw:
        result["raw_predictions"] = predictions

    model.eval()  # Set model back to evaluation mode
    return result
