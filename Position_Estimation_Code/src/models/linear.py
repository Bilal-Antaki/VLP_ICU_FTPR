from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def build_linear_model(model_type='linear', **kwargs):
    """
    Build different types of linear models
    
    Args:
        model_type: 'linear', 'ridge', 'lasso', 'elastic', 'poly'
        **kwargs: model-specific parameters
    """
    if model_type == 'linear':
        return LinearRegression(**kwargs)
    
    elif model_type == 'ridge':
        alpha = kwargs.get('alpha', 1.0)
        return Ridge(alpha=alpha, **{k: v for k, v in kwargs.items() if k != 'alpha'})
    
    elif model_type == 'lasso':
        alpha = kwargs.get('alpha', 1.0)
        return Lasso(alpha=alpha, max_iter=2000, **{k: v for k, v in kwargs.items() if k != 'alpha'})
    
    elif model_type == 'elastic':
        alpha = kwargs.get('alpha', 1.0)
        l1_ratio = kwargs.get('l1_ratio', 0.5)
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, 
                         **{k: v for k, v in kwargs.items() if k not in ['alpha', 'l1_ratio']})
    
    elif model_type == 'poly':
        degree = kwargs.get('degree', 2)
        alpha = kwargs.get('alpha', 1.0)
        reg_type = kwargs.get('reg_type', 'ridge')
        
        # Create polynomial pipeline
        if reg_type == 'ridge':
            regressor = Ridge(alpha=alpha)
        elif reg_type == 'lasso':
            regressor = Lasso(alpha=alpha, max_iter=2000)
        else:
            regressor = LinearRegression()
            
        return Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('regressor', regressor)
        ])
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# Backwards compatibility
def build_linear_model_simple(**kwargs):
    """Simple linear regression for backwards compatibility"""
    return LinearRegression(**kwargs)