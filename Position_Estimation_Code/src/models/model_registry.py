# src/models/model_registry.py
from .linear import build_linear_model, build_linear_model_simple
from .svr import build_svr_model, build_svr_optimized, build_svr_linear, build_svr_poly
from .lstm import build_lstm_model

MODEL_REGISTRY = {
    # Linear models
    "linear": build_linear_model_simple,
    "ridge": lambda **kwargs: build_linear_model(model_type='ridge', **kwargs),
    "lasso": lambda **kwargs: build_linear_model(model_type='lasso', **kwargs),
    "elastic": lambda **kwargs: build_linear_model(model_type='elastic', **kwargs),
    "poly": lambda **kwargs: build_linear_model(model_type='poly', **kwargs),
    
    # SVR models
    "svr": build_svr_optimized,
    "svr_linear": build_svr_linear,
    "svr_poly": build_svr_poly,
    "svr_rbf": lambda **kwargs: build_svr_model(kernel='rbf', **kwargs),
    
    # Deep learning models
    "lstm": build_lstm_model,
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
        available_models = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Model '{name}' not found in registry. Available models: {available_models}")
    
    return MODEL_REGISTRY[name](**kwargs)

def list_available_models():
    """List all available models grouped by category"""
    categories = {
        'Linear': ['linear', 'ridge', 'lasso', 'elastic', 'poly'],
        'SVM': ['svr', 'svr_linear', 'svr_poly', 'svr_rbf'],
        'Deep Learning': ['lstm']
    }
    
    print("Available Models by Category:")
    print("=" * 50)
    for category, models in categories.items():
        print(f"\n{category}:")
        for model in models:
            if model in MODEL_REGISTRY:
                print(f"  - {model}")
    
    return list(MODEL_REGISTRY.keys())