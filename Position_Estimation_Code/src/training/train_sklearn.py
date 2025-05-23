from src.models.model_registry import get_model, list_available_models
from src.data.loader import load_cir_data, extract_features_and_target
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def train_single_model(model_name: str, X_train, X_test, y_train, y_test, **model_kwargs):
    """Train a single sklearn model and return RMSE metrics only"""
    try:
        model = get_model(model_name, **model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate RMSE only
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Cross-validation RMSE
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
        except:
            cv_rmse = None
        
        return {
            'model': model,
            'y_pred': y_pred,
            'rmse': rmse,
            'cv_rmse': cv_rmse,
            'pred_std': np.std(y_pred),
            'pred_range': (np.min(y_pred), np.max(y_pred))
        }
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return None

def train_linear_on_all(processed_dir: str):
    """Original function for backwards compatibility"""
    df = load_cir_data(processed_dir, filter_keyword="FCPR-D1")
    X, y = extract_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = get_model("linear")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Linear Regression RMSE: {rmse:.4f}")
    
    return y_test.tolist(), y_pred.tolist()

def train_all_sklearn_models(processed_dir: str, test_size: float = 0.2):
    """
    Train and compare multiple sklearn models - RMSE only
    """
    print("Loading data...")
    df = load_cir_data(processed_dir, filter_keyword="FCPR-D1")
    X, y = extract_features_and_target(df)
    
    print(f"Dataset shape: {X.shape}, Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to test
    models_to_test = [
        ('Linear', 'linear', {}, X_train, X_test),
        ('Ridge', 'ridge', {'alpha': 1.0}, X_train_scaled, X_test_scaled),
        ('Ridge (α=10)', 'ridge', {'alpha': 10.0}, X_train_scaled, X_test_scaled),
        ('Lasso', 'lasso', {'alpha': 0.1}, X_train_scaled, X_test_scaled),
        ('Elastic Net', 'elastic', {'alpha': 0.1, 'l1_ratio': 0.5}, X_train_scaled, X_test_scaled),
        ('Polynomial (deg=2)', 'poly', {'degree': 2}, X_train, X_test),
        ('SVR RBF', 'svr', {}, X_train, X_test),  # Already includes scaling
        ('SVR Linear', 'svr_linear', {'C': 10.0}, X_train, X_test),
        ('SVR Poly', 'svr_poly', {'degree': 2, 'C': 10.0}, X_train, X_test),
    ]
    
    results = []
    
    print(f"\nTraining {len(models_to_test)} models...")
    print("-" * 80)
    
    for display_name, model_name, kwargs, X_tr, X_te in models_to_test:
        print(f"Training {display_name}...")
        result = train_single_model(model_name, X_tr, X_te, y_train, y_test, **kwargs)
        
        if result:
            result['name'] = display_name
            result['model_type'] = model_name
            results.append(result)
            
            print(f"  RMSE: {result['rmse']:.4f}")
            if result['cv_rmse']:
                print(f"  CV-RMSE: {result['cv_rmse']:.4f}")
            print(f"  Pred range: [{result['pred_range'][0]:.2f}, {result['pred_range'][1]:.2f}]")
        print()
    
    # Create results summary - RMSE only
    if results:
        results_df = pd.DataFrame([
            {
                'Model': r['name'],
                'RMSE': r['rmse'],
                'CV-RMSE': r['cv_rmse'] if r['cv_rmse'] else np.nan,
                'Pred_Std': r['pred_std'],
                'Pred_Range': f"[{r['pred_range'][0]:.1f}, {r['pred_range'][1]:.1f}]"
            }
            for r in results
        ]).round(4)
        
        print("=" * 80)
        print("MODEL COMPARISON SUMMARY (RMSE ONLY)")
        print("=" * 80)
        print(results_df.to_string(index=False))
        
        # Find best model by RMSE
        best_rmse = results_df.loc[results_df['RMSE'].idxmin()]
        
        print(f"\nBest RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
        
        return results, results_df
    
    return None, None

def get_best_sklearn_model(processed_dir: str):
    """Get the best performing sklearn model by RMSE"""
    results, _ = train_all_sklearn_models(processed_dir)
    if results:
        best_result = min(results, key=lambda x: x['rmse'])
        return best_result['model'], best_result
    return None, None