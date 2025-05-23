from .linear import build_linear_model, build_linear_model_simple
from .svr import build_svr_model, build_svr_optimized, build_svr_linear, build_svr_poly
#from .knn import build_knn_model
#from .mlp import build_mlp_model

# Optional: import DL models if needed
from .lstm import build_lstm_model
# from .rnn import build_rnn_model

MODEL_REGISTRY = {
    # Linear models
    "linear": build_linear_model_simple,  # Keep simple for backwards compatibility
    "linear_advanced": build_linear_model,
    "ridge": lambda **kwargs: build_linear_model(model_type='ridge', **kwargs),
    "lasso": lambda **kwargs: build_linear_model(model_type='lasso', **kwargs),
    "elastic": lambda **kwargs: build_linear_model(model_type='elastic', **kwargs),
    "poly": lambda **kwargs: build_linear_model(model_type='poly', **kwargs),
    
    # SVR models
    "svr": build_svr_optimized,  # Use optimized version as default
    "svr_basic": build_svr_model,
    "svr_linear": build_svr_linear,
    "svr_poly": build_svr_poly,
    "svr_rbf": lambda **kwargs: build_svr_model(kernel='rbf', **kwargs),
    
    # Deep learning models
    "lstm": build_lstm_model,
    
    # Placeholders for future models
    #"knn": build_knn_model,
    #"mlp": build_mlp_model,
    # "rnn": build_rnn_model,
}

def get_model(name: str, **kwargs):
    """
    Get a model from the registry
    
    Args:
        name: Model name from MODEL_REGISTRY
        **kwargs: Model-specific parameters
        
    Returns:
        Configured model instance
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        available_models = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found in registry. Available models: {available_models}")
    
    return MODEL_REGISTRY[name](**kwargs)

def list_available_models():
    """List all available models"""
    return list(MODEL_REGISTRY.keys())