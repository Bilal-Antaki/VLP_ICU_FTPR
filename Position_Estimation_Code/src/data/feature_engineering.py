# src/data/feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

def create_engineered_features(df, features=['PL', 'RMS'], include_coordinates=False, include_categorical=True):
    """
    Create engineered features for better model performance
    
    Args:
        df: Input DataFrame with at least PL and RMS columns
        features: Base features to use
        include_coordinates: Whether to include X, Y coordinates as features
        include_categorical: Whether to include categorical interaction features
        
    Returns:
        DataFrame with engineered features
    """
    feature_df = df.copy()
    
    # 1. Basic features
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
    
    # 2. Coordinate-based features if available
    if include_coordinates and 'X' in df.columns and 'Y' in df.columns:
        # Polar coordinates
        feature_df['radius'] = np.sqrt(df['X']**2 + df['Y']**2)
        feature_df['angle'] = np.arctan2(df['Y'], df['X'])
        
        # Distance from origin
        feature_df['manhattan_dist'] = np.abs(df['X']) + np.abs(df['Y'])
        
        # Quadrant information
        feature_df['quadrant'] = (
            (df['X'] >= 0).astype(int) * 2 + 
            (df['Y'] >= 0).astype(int)
        )
        
        # Coordinate ratios
        feature_df['X_Y_ratio'] = df['X'] / (df['Y'] + 1e-10)
        feature_df['Y_X_ratio'] = df['Y'] / (df['X'] + 1e-10)
        
        # Coordinate products
        feature_df['X_Y_product'] = df['X'] * df['Y']
        
    # 3. Statistical features (if we have grouped data)
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
    
    # 4. Domain-specific features for wireless propagation
    if 'PL' in df.columns:
        # Free space path loss at 2.4 GHz reference
        if 'radius' in feature_df.columns:
            freq_ghz = 2.4  # Assumed frequency
            feature_df['FSPL_2.4GHz'] = 20 * np.log10(feature_df['radius'] + 1e-10) + 20 * np.log10(freq_ghz * 1e9) - 147.55
            feature_df['PL_excess'] = df['PL'] - feature_df['FSPL_2.4GHz']
    
    # 5. Interaction features (only if requested)
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
            # Skip if binning fails (e.g., too few unique values)
            pass
    
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

def create_polynomial_features(X, degree=2, include_bias=False):
    """
    Create polynomial features
    
    Args:
        X: Input features
        degree: Polynomial degree
        include_bias: Whether to include bias term
        
    Returns:
        Polynomial features array and feature names
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_poly = poly.fit_transform(X)
    
    # Get feature names
    feature_names = poly.get_feature_names_out(X.columns if hasattr(X, 'columns') else None)
    
    return X_poly, feature_names

def create_distance_features(df):
    """
    Create distance-based features specifically for position estimation
    
    Args:
        df: DataFrame with position data
        
    Returns:
        DataFrame with distance features
    """
    feature_df = pd.DataFrame()
    
    if 'X' in df.columns and 'Y' in df.columns:
        # Various distance metrics
        feature_df['euclidean_dist'] = np.sqrt(df['X']**2 + df['Y']**2)
        feature_df['manhattan_dist'] = np.abs(df['X']) + np.abs(df['Y'])
        feature_df['chebyshev_dist'] = np.maximum(np.abs(df['X']), np.abs(df['Y']))
        feature_df['canberra_dist'] = (np.abs(df['X']) + np.abs(df['Y'])) / (np.abs(df['X']) + np.abs(df['Y']) + 1e-10)
        
        # Log-distance features
        feature_df['log_euclidean'] = np.log1p(feature_df['euclidean_dist'])
        feature_df['log_manhattan'] = np.log1p(feature_df['manhattan_dist'])
        
        # Normalized coordinates
        max_coord = np.maximum(np.abs(df['X']).max(), np.abs(df['Y']).max())
        feature_df['X_normalized'] = df['X'] / (max_coord + 1e-10)
        feature_df['Y_normalized'] = df['Y'] / (max_coord + 1e-10)
        
    return feature_df

def create_lag_features(df, features=['PL', 'RMS'], lags=[1, 2, 3]):
    """
    Create lag features for time-series like data
    
    Args:
        df: Input DataFrame (should be sorted by some order)
        features: Features to create lags for
        lags: List of lag values
        
    Returns:
        DataFrame with lag features
    """
    feature_df = df.copy()
    
    for feature in features:
        if feature in df.columns:
            for lag in lags:
                feature_df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
                
            # Rolling statistics
            for window in [3, 5]:
                feature_df[f'{feature}_rolling_mean_{window}'] = (
                    df[feature].rolling(window=window, center=True).mean()
                )
                feature_df[f'{feature}_rolling_std_{window}'] = (
                    df[feature].rolling(window=window, center=True).std()
                )
    
    # Fill NaN values
    feature_df = feature_df.fillna(method='bfill').fillna(method='ffill')
    
    return feature_df