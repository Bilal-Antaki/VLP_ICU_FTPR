# src/models/ensemble.py
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    VotingRegressor,
    BaggingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def build_random_forest(n_estimators=100, max_depth=None, min_samples_split=2, 
                       min_samples_leaf=1, max_features='sqrt', **kwargs):
    """
    Build Random Forest Regressor
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples to split internal node
        min_samples_leaf: Minimum samples in leaf node
        max_features: Number of features to consider for best split
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
        **kwargs
    )

def build_gradient_boosting(n_estimators=100, learning_rate=0.1, max_depth=3,
                          subsample=1.0, loss='squared_error', **kwargs):
    """
    Build Gradient Boosting Regressor
    
    Args:
        n_estimators: Number of boosting stages
        learning_rate: Shrinks contribution of each tree
        max_depth: Maximum depth of individual trees
        subsample: Fraction of samples for fitting base learners
        loss: Loss function ('squared_error', 'absolute_error', 'huber')
    """
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        loss=loss,
        random_state=42,
        **kwargs
    )

def build_extra_trees(n_estimators=100, max_depth=None, min_samples_split=2, **kwargs):
    """Build Extra Trees Regressor (more random than Random Forest)"""
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1,
        **kwargs
    )

def build_adaboost(n_estimators=50, learning_rate=1.0, loss='linear', **kwargs):
    """
    Build AdaBoost Regressor
    
    Args:
        n_estimators: Number of weak learners
        learning_rate: Weight applied to each classifier
        loss: Loss function ('linear', 'square', 'exponential')
    """
    return AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        loss=loss,
        random_state=42,
        **kwargs
    )

def build_voting_ensemble(**kwargs):
    """
    Build Voting Regressor combining multiple models
    
    Combines predictions from multiple models by averaging
    """
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('ridge', Ridge(alpha=1.0)),
        ('svr', SVR(kernel='rbf', C=100)),
        ('knn', KNeighborsRegressor(n_neighbors=7, weights='distance'))
    ]
    
    return VotingRegressor(estimators=estimators, **kwargs)

def build_bagging(base_estimator=None, n_estimators=10, max_samples=1.0, 
                 max_features=1.0, **kwargs):
    """
    Build Bagging Regressor
    
    Args:
        base_estimator: Base estimator (default: DecisionTreeRegressor)
        n_estimators: Number of base estimators
        max_samples: Number of samples to draw
        max_features: Number of features to draw
    """
    if base_estimator is None:
        base_estimator = DecisionTreeRegressor()
        
    return BaggingRegressor(
        estimator=base_estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
        **kwargs
    )

# XGBoost and LightGBM wrappers (if installed)
try:
    import xgboost as xgb
    
    def build_xgboost(n_estimators=100, max_depth=6, learning_rate=0.3,
                     subsample=1.0, colsample_bytree=1.0, **kwargs):
        """Build XGBoost Regressor"""
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
except ImportError:
    def build_xgboost(**kwargs):
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    
    def build_lightgbm(n_estimators=100, num_leaves=31, learning_rate=0.1,
                      feature_fraction=1.0, bagging_fraction=1.0, **kwargs):
        """Build LightGBM Regressor"""
        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
except ImportError:
    def build_lightgbm(**kwargs):
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")