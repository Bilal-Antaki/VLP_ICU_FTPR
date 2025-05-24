# src/training/train_enhanced.py
from src.models.model_registry import get_model, list_available_models
from src.data.loader import load_cir_data, extract_features_and_target
from src.evaluation.metrics import calculate_all_metrics, compare_models, print_metrics_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

def train_model_with_metrics(model_name, X_train, X_test, y_train, y_test, **model_kwargs):
    """Train a model and calculate comprehensive metrics"""
    start_time = time.time()
    
    try:
        # Get and train model
        model = get_model(model_name, **model_kwargs)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Training time
        train_time = time.time() - start_time
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_test, y_pred, model_name)
        metrics['train_time'] = train_time
        
        # Cross-validation if model supports it
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error', n_jobs=-1)
            metrics['cv_rmse'] = np.sqrt(-cv_scores.mean())
            metrics['cv_std'] = np.sqrt(cv_scores.std())
        except:
            metrics['cv_rmse'] = np.nan
            metrics['cv_std'] = np.nan
        
        return {
            'success': True,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,  # Add y_test to results
            'metrics': metrics,
            'name': model_name
        }
        
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'name': model_name
        }

def hyperparameter_search(model_name, X_train, y_train, param_grid, cv=5):
    """Perform hyperparameter search for a model"""
    base_model = get_model(model_name)
    
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': np.sqrt(-grid_search.best_score_),
        'model': grid_search.best_estimator_
    }

def train_all_models_enhanced(processed_dir: str, test_size: float = 0.2, 
                            include_slow_models: bool = True,
                            include_deep_learning: bool = False):
    """
    Train and compare all available models with comprehensive metrics
    
    Args:
        processed_dir: Directory with processed data
        test_size: Test set size
        include_slow_models: Include computationally expensive models
        include_deep_learning: Include deep learning models (requires separate handling)
    """
    print("Enhanced Model Training and Evaluation")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    df = load_cir_data(processed_dir, filter_keyword="FCPR-D1")
    X, y = extract_features_and_target(df)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to test with configurations
    model_configs = [
        # Linear models
        ('Linear Regression', 'linear', {}, False),
        ('Ridge (α=0.1)', 'ridge', {'alpha': 0.1}, True),
        ('Ridge (α=1.0)', 'ridge', {'alpha': 1.0}, True),
        ('Ridge (α=10)', 'ridge', {'alpha': 10.0}, True),
        ('Lasso (α=0.01)', 'lasso', {'alpha': 0.01}, True),
        ('Lasso (α=0.1)', 'lasso', {'alpha': 0.1}, True),
        ('ElasticNet', 'elastic', {'alpha': 0.1, 'l1_ratio': 0.5}, True),
        ('Polynomial (deg=2)', 'poly', {'degree': 2}, False),
        ('Polynomial (deg=3)', 'poly', {'degree': 3}, False),
        
        # SVM models
        ('SVR RBF', 'svr', {}, False),
        ('SVR Linear', 'svr_linear', {'C': 10.0}, False),
        ('SVR Poly', 'svr_poly', {'degree': 2, 'C': 10.0}, False),
    ]
    
    # Try to add optional models
    from src.models.model_registry import MODEL_REGISTRY
    
    if 'knn' in MODEL_REGISTRY:
        model_configs.extend([
            ('KNN (k=5)', 'knn', {'n_neighbors': 5}, True),
            ('KNN (k=7)', 'knn', {'n_neighbors': 7}, True),
            ('KNN (k=10)', 'knn', {'n_neighbors': 10}, True),
            ('KNN Distance Weighted', 'knn_distance', {'n_neighbors': 7}, True),
        ])
    
    if 'mlp' in MODEL_REGISTRY:
        model_configs.extend([
            ('MLP (100,50)', 'mlp', {'hidden_layers': (100, 50)}, True),
            ('MLP (64,32,16)', 'mlp', {'hidden_layers': (64, 32, 16)}, True),
        ])
    
    if 'random_forest' in MODEL_REGISTRY:
        model_configs.extend([
            ('Random Forest (100)', 'random_forest', {'n_estimators': 100}, False),
            ('Random Forest (200)', 'random_forest', {'n_estimators': 200}, False),
        ])
    
    if 'gradient_boosting' in MODEL_REGISTRY:
        model_configs.append(('Gradient Boosting', 'gradient_boosting', {'n_estimators': 100}, False))
        
    if 'extra_trees' in MODEL_REGISTRY:
        model_configs.append(('Extra Trees', 'extra_trees', {'n_estimators': 100}, False))
    
    if 'adaboost' in MODEL_REGISTRY:
        model_configs.append(('AdaBoost', 'adaboost', {'n_estimators': 50}, False))
    
    # Add slow models if requested
    if include_slow_models:
        model_configs.extend([
            ('Random Forest (500)', 'random_forest', {'n_estimators': 500}, False),
            ('GB (200 trees)', 'gradient_boosting', {'n_estimators': 200}, False),
            ('Voting Ensemble', 'voting', {}, False),
            ('Bagging', 'bagging', {'n_estimators': 50}, False),
        ])
        
        # Try to add XGBoost and LightGBM
        try:
            model_configs.append(('XGBoost', 'xgboost', {'n_estimators': 100}, False))
        except:
            print("XGBoost not available")
            
        try:
            model_configs.append(('LightGBM', 'lightgbm', {'n_estimators': 100}, False))
        except:
            print("LightGBM not available")
    
    # Train all models
    results = []
    successful_results = []
    
    print(f"\nTraining {len(model_configs)} model configurations...")
    print("-" * 60)
    
    for display_name, model_name, kwargs, needs_scaling in model_configs:
        print(f"\nTraining {display_name}...", end=' ')
        
        # Use scaled or unscaled data
        X_tr = X_train_scaled if needs_scaling else X_train
        X_te = X_test_scaled if needs_scaling else X_test
        
        result = train_model_with_metrics(
            model_name, X_tr, X_te, y_train, y_test, **kwargs
        )
        
        if result['success']:
            print(f"✓ RMSE: {result['metrics']['rmse']:.4f}, Time: {result['metrics']['train_time']:.2f}s")
            results.append(result)
            successful_results.append({
                'name': display_name,
                'y_true': y_test,
                'y_pred': result['y_pred']
            })
        else:
            print(f"✗ Failed: {result['error']}")
    
    if not results:
        print("\nNo models trained successfully!")
        return None, None
    
    # Create comparison DataFrame
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    metrics_list = [r['metrics'] for r in results]
    comparison_df = pd.DataFrame(metrics_list)
    
    # Select columns to display
    display_columns = [
        'model_name', 'rmse', 'mae', 'r2', 'mape', 
        'median_abs_error', 'p90_error', 'max_error',
        'cv_rmse', 'train_time'
    ]
    
    display_df = comparison_df[display_columns].round(4)
    display_df = display_df.sort_values('rmse')
    
    print(display_df.to_string(index=False))
    
    # Best models summary
    print("\n" + "=" * 60)
    print("TOP 5 MODELS BY RMSE")
    print("=" * 60)
    
    top_5 = display_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"\n{row['model_name']}:")
        print(f"  RMSE: {row['rmse']:.4f}")
        print(f"  MAE:  {row['mae']:.4f}")
        print(f"  R²:   {row['r2']:.4f}")
        print(f"  MAPE: {row['mape']:.2f}%")
        if not pd.isna(row['cv_rmse']):
            print(f"  CV-RMSE: {row['cv_rmse']:.4f}")
    
    # Statistical comparison
    best_model = display_df.iloc[0]
    print(f"\nBest Model: {best_model['model_name']}")
    print(f"RMSE: {best_model['rmse']:.4f}")
    
    # Find best model object
    best_model_obj = None
    for r in results:
        if r['metrics']['model_name'] == best_model['model_name']:
            best_model_obj = r['model']
            break
    
    return results, comparison_df, best_model_obj

def automated_model_selection(processed_dir: str, metric='rmse'):
    """
    Automatically select the best model based on cross-validation
    
    Args:
        processed_dir: Directory with processed data
        metric: Metric to optimize ('rmse', 'mae', 'r2')
    """
    results, comparison_df, best_model = train_all_models_enhanced(
        processed_dir, 
        include_slow_models=True
    )
    
    if results is None:
        return None
    
    print(f"\n{'='*60}")
    print(f"AUTOMATED MODEL SELECTION")
    print(f"{'='*60}")
    print(f"Best model based on {metric}: {comparison_df.iloc[0]['model_name']}")
    print(f"Performance: {comparison_df.iloc[0][metric]:.4f}")
    
    return best_model