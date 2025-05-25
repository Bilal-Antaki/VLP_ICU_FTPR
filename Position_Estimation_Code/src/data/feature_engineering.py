# src/data/feature_engineering.py
import numpy as np
import pandas as pd

def create_engineered_features(df, features=['PL', 'RMS'], include_categorical=True):
    """
    Create engineered features for better model performance
    
    Args:
        df: Input DataFrame with at least PL and RMS columns
        features: Base features to use (ONLY PL and RMS)
        include_coordinates: DEPRECATED - always False
        include_categorical: Whether to include categorical interaction features
        
    Returns:
        DataFrame with engineered features
    """
    feature_df = df.copy()
    
    # use PL and RMS features
    if 'PL' in df.columns and 'RMS' in df.columns:
        # Ratio features
        feature_df['PL_RMS_ratio'] = df['PL'] / (df['RMS'] + 1e-10)
        feature_df['RMS_PL_ratio'] = df['RMS'] / (df['PL'] + 1e-10)
        
        # Product features
        feature_df['PL_RMS_product'] = df['PL'] * df['RMS']
        
        # Difference features
        feature_df['PL_RMS_diff'] = df['PL'] - df['RMS']
        feature_df['PL_RMS_abs_diff'] = np.abs(df['PL'] - df['RMS'])
        
        # Power features
        feature_df['PL_squared'] = df['PL'] ** 2
        feature_df['RMS_squared'] = df['RMS'] ** 2
        feature_df['PL_sqrt'] = np.sqrt(np.abs(df['PL']))
        feature_df['RMS_sqrt'] = np.sqrt(np.abs(df['RMS']))
        
        # Log features (handle negative values)
        feature_df['PL_log'] = np.log1p(np.abs(df['PL']))
        feature_df['RMS_log'] = np.log1p(np.abs(df['RMS']))
        
        # Exponential features (scaled to prevent overflow)
        feature_df['PL_exp'] = np.exp(df['PL'] / 100)
        feature_df['RMS_exp'] = np.exp(df['RMS'] / 10)
    
    # Statistical features (if we have grouped data)
    if 'source_file' in df.columns:
        # Add group statistics
        for feature in features:
            if feature in df.columns:
                group_stats = df.groupby('source_file')[feature].agg(['mean', 'std', 'min', 'max'])
                feature_df[f'{feature}_group_mean'] = df['source_file'].map(group_stats['mean'])
                feature_df[f'{feature}_group_std'] = df['source_file'].map(group_stats['std'])
                feature_df[f'{feature}_normalized'] = (
                    (df[feature] - feature_df[f'{feature}_group_mean']) / 
                    (feature_df[f'{feature}_group_std'] + 1e-10)
                )
    
    # Interaction features (only if requested)
    if include_categorical and 'PL' in df.columns and 'RMS' in df.columns:
        # Binned interactions
        try:
            pl_bins = pd.qcut(df['PL'], q=5, labels=['VL', 'L', 'M', 'H', 'VH'])
            rms_bins = pd.qcut(df['RMS'], q=5, labels=['VL', 'L', 'M', 'H', 'VH'])
            feature_df['PL_RMS_interaction'] = pl_bins.astype(str) + '_' + rms_bins.astype(str)
            
            # Convert to dummy variables
            interaction_dummies = pd.get_dummies(feature_df['PL_RMS_interaction'], prefix='interaction')
            feature_df = pd.concat([feature_df, interaction_dummies], axis=1)
            
            # Drop the original categorical column
            feature_df = feature_df.drop('PL_RMS_interaction', axis=1)
        except:
            # Skip if binning fails
            pass
    
    # Remove any coordinate-based features if they exist
    coordinate_cols = ['X', 'Y', 'radius', 'angle', 'manhattan_dist', 'quadrant', 
                      'X_Y_ratio', 'Y_X_ratio', 'X_Y_product', 'X_normalized', 'Y_normalized']
    for col in coordinate_cols:
        if col in feature_df.columns:
            feature_df = feature_df.drop(col, axis=1)
    
    return feature_df

def select_features(X, y, method='correlation', threshold=0.1):
    """
    Select relevant features based on correlation or importance
    
    Args:
        X: Feature DataFrame
        y: Target values
        method: 'correlation' or 'mutual_info'
        threshold: Threshold for feature selection
        
    Returns:
        List of selected feature names
    """
    # First, identify numeric columns only
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove any coordinate-based features
    coordinate_cols = ['X', 'Y', 'radius', 'angle', 'manhattan_dist', 'quadrant',
                      'X_Y_ratio', 'Y_X_ratio', 'X_Y_product', 'X_normalized', 'Y_normalized']
    numeric_columns = [col for col in numeric_columns if col not in coordinate_cols]
    
    X_numeric = X[numeric_columns]
    
    if len(numeric_columns) == 0:
        raise ValueError("No numeric features found in X")
    
    if method == 'correlation':
        # Calculate correlation with target (numeric features only)
        correlations = X_numeric.corrwith(pd.Series(y)).abs()
        selected_features = correlations[correlations > threshold].index.tolist()
        
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(X_numeric, y)
        mi_df = pd.DataFrame({'feature': X_numeric.columns, 'mi_score': mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)
        selected_features = mi_df[mi_df['mi_score'] > threshold]['feature'].tolist()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Always include original features
    base_features = ['PL', 'RMS']
    for feat in base_features:
        if feat in X.columns and feat not in selected_features:
            selected_features.append(feat)
    
    return selected_features