# src/models/model_registry.py
from .linear import build_linear_model, build_linear_model_simple
from .svr import build_svr_model, build_svr_optimized, build_svr_linear, build_svr_poly
from .knn import build_knn_model, build_knn_optimized
from .mlp import build_mlp_model, build_mlp_sklearn, build_mlp_torch
from .ensemble import (
    build_random_forest, build_gradient_boosting, build_extra_trees,
    build_adaboost, build_voting_ensemble, build_bagging,
    build_xgboost, build_lightgbm
)

# Optional: import DL models
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
    
    # KNN models
    "knn": build_knn_optimized,
    "knn_basic": build_knn_model,
    "knn_uniform": lambda **kwargs: build_knn_model(weights='uniform', **kwargs),
    "knn_distance": lambda **kwargs: build_knn_model(weights='distance', **kwargs),
    
    # Neural Networks
    "mlp": build_mlp_model,
    "mlp_sklearn": build_mlp_sklearn,
    "mlp_torch": build_mlp_torch,
    
    # Tree-based Ensemble models
    "random_forest": build_random_forest,
    "rf": build_random_forest,  # Alias
    "gradient_boosting": build_gradient_boosting,
    "gb": build_gradient_boosting,  # Alias
    "extra_trees": build_extra_trees,
    "et": build_extra_trees,  # Alias
    "adaboost": build_adaboost,
    "ada": build_adaboost,  # Alias
    
    # Advanced Ensemble models
    "voting": build_voting_ensemble,
    "bagging": build_bagging,
    "xgboost": build_xgboost,
    "xgb": build_xgboost,  # Alias
    "lightgbm": build_lightgbm,
    "lgb": build_lightgbm,  # Alias
    
    # Deep learning models
    "lstm": build_lstm_model,
    
    # Placeholders for future models
    # "rnn": build_rnn_model,
    # "gru": build_gru_model,
    # "transformer": build_transformer_model,
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
        'Neighbors': ['knn', 'knn_uniform', 'knn_distance'],
        'Neural Networks': ['mlp', 'mlp_sklearn'],
        'Tree Ensembles': ['random_forest', 'gradient_boosting', 'extra_trees', 'adaboost'],
        'Advanced Ensembles': ['voting', 'bagging', 'xgboost', 'lightgbm'],
        'Deep Learning': ['lstm', 'mlp_torch']
    }
    
    print("Available Models by Category:")
    print("=" * 50)
    for category, models in categories.items():
        print(f"\n{category}:")
        for model in models:
            if model in MODEL_REGISTRY:
                print(f"  - {model}")
    
    return list(MODEL_REGISTRY.keys())