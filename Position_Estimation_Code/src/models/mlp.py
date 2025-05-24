import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# PyTorch MLP implementation
class MLPRegressorTorch(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[64, 32], dropout=0.2, activation='relu'):
        super(MLPRegressorTorch, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
                
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)

# Sklearn MLP wrapper for compatibility
def build_mlp_sklearn(hidden_layers=(100, 50), activation='relu', solver='adam', 
                     alpha=0.0001, learning_rate='adaptive', max_iter=1000, **kwargs):
    """
    Build sklearn MLPRegressor with scaling
    
    Args:
        hidden_layers: Tuple of hidden layer sizes
        activation: Activation function ('relu', 'tanh', 'logistic')
        solver: Weight optimization solver ('adam', 'sgd', 'lbfgs')
        alpha: L2 penalty parameter
        learning_rate: Learning rate schedule ('constant', 'invscaling', 'adaptive')
        max_iter: Maximum iterations
        **kwargs: Additional MLPRegressor parameters
    """
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=42,
            **kwargs
        ))
    ])

def build_mlp_model(**kwargs):
    """Default MLP model builder"""
    return build_mlp_sklearn(**kwargs)

def build_mlp_torch(**kwargs):
    """Build PyTorch MLP model"""
    return MLPRegressorTorch(**kwargs)