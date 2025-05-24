from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_knn_model(n_neighbors=5, weights='uniform', metric='euclidean', **kwargs):
    """
    Build KNN regressor with various distance metrics
    
    Args:
        n_neighbors: Number of neighbors to use
        weights: 'uniform' or 'distance' - weight function used in prediction
        metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
        **kwargs: Additional KNeighborsRegressor parameters
    """
    
    # KNN benefits from scaling
    return Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            **kwargs
        ))
    ])

def build_knn_optimized(**kwargs):
    """Build KNN with optimized parameters for position estimation"""
    return build_knn_model(
        n_neighbors=7,
        weights='distance',  # Weight by inverse distance
        metric='euclidean',
        **kwargs
    )