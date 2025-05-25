from sklearn.linear_model import LinearRegression

def build_linear_model(model_type='linear', **kwargs):
    """
    Args:
        model_type: 'linear'
        **kwargs: model-specific parameters
    """
    if model_type == 'linear':
        return LinearRegression(**kwargs)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# Backwards compatibility
def build_linear_model_simple(**kwargs):
    """Simple linear regression for backwards compatibility"""
    return LinearRegression(**kwargs)