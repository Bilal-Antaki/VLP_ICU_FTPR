import numpy as np

def compute_rmse(y_true, y_pred):
    """Compute root mean squared error between true and predicted values."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))