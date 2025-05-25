# main.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.training.train_sklearn import train_all_models_enhanced, automated_model_selection
from src.training.train_dl import train_lstm_on_all
from src.data.loader import load_cir_data, extract_features_and_target
from src.data.feature_engineering import create_engineered_features, select_features
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def run_analysis():
    """Run analysis with Linear, SVR, and LSTM models"""
    processed_dir = "data/processed"
    
    print("=" * 80)
    print("COMPREHENSIVE POSITION ESTIMATION ANALYSIS")
    print("=" * 80)
    
    # 1. Load and explore data
    print("\n1. Loading and exploring data...")
    df_list = []
    
    # Load all available datasets
    for keyword in ['FCPR-D1']:
        try:
            df_temp = load_cir_data(processed_dir, filter_keyword=keyword)
            print(f"  Loaded {keyword}: {len(df_temp)} samples")
            df_list.append(df_temp)
        except:
            print(f"  {keyword} not found")
    
    # Combine all data
    df_all = pd.concat(df_list, ignore_index=True) if df_list else None
    
    if df_all is None:
        print("No data found!")
        return
    
    print(f"\nTotal samples: {len(df_all)}")
    print(f"Unique sources: {df_all['source_file'].nunique()}")
    
    # 2. Feature Engineering
    print("\n2. Feature Engineering (PL and RMS features only)...")
    df_engineered = create_engineered_features(df_all, include_categorical=False)
    
    # Select features - exclude any coordinate-based features
    feature_cols = [col for col in df_engineered.columns 
                   if col not in ['r', 'X', 'Y', 'source_file', 'radius', 'angle', 
                                 'manhattan_dist', 'quadrant', 'X_Y_ratio', 'Y_X_ratio', 
                                 'X_Y_product', 'X_normalized', 'Y_normalized']]
    
    X = df_engineered[feature_cols]
    y = df_engineered['r']
    
    # Select best features
    selected_features = select_features(X, y, method='correlation', threshold=0.3)
    print(f"  Selected {len(selected_features)} features from {len(feature_cols)} total")
    print(f"  Top features: {selected_features[:10]}")
    
    # 3. Train models with basic features
    print("\n3. Training models with BASIC features (PL, RMS only)...")
    results_basic, df_basic, _ = train_all_models_enhanced(
        processed_dir, 
        include_slow_models=False
    )
    
    # 4. Deep Learning Comparison
    print("\n4. Training LSTM model...")
    lstm_results = train_lstm_on_all(processed_dir)
    
    # 5. Create visualization figures
    #create_analysis_figures(results_basic, lstm_results, df_all)
    
    # 6. Statistical Analysis
    print("\n6. Statistical Analysis...")
    perform_statistical_analysis(results_basic, lstm_results)
    
    # 7. Save results
    save_analysis_results(results_basic, lstm_results, df_basic)

def create_analysis_figures(sklearn_results, lstm_results, df_raw):
    """Create separate figure windows for different visualizations"""
    
    # Figure 1: Data Exploration
    fig1, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle('Data Exploration', fontsize=16)
    
    # Correlation Heatmap
    correlation_matrix = df_raw[['PL', 'RMS', 'r']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax[0])
    ax[0].set_title('Feature Correlation Matrix')
    
    # PL vs Distance
    scatter = ax[1].scatter(df_raw['r'], df_raw['PL'], c=df_raw['RMS'], cmap='viridis', alpha=0.6, s=20)
    ax[1].set_xlabel('Distance (r)')
    ax[1].set_ylabel('Path Loss (PL)')
    ax[1].set_title('PL vs Distance (colored by RMS)')
    fig1.colorbar(scatter, ax=ax[1], label='RMS')
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Model Performance
    fig2, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Model Performance Comparison', fontsize=16)
    
    # Model Performance Comparison
    if sklearn_results:
        model_names = [r['name'] for r in sklearn_results[:5] if r['success']]
        rmse_values = [r['metrics']['rmse'] for r in sklearn_results[:5] if r['success']]
        model_names.append('LSTM')
        rmse_values.append(lstm_results['rmse'])
        
        bars = ax[0].barh(model_names, rmse_values)
        ax[0].set_xlabel('RMSE')
        ax[0].set_title('Model Performance by RMSE')
        bars[np.argmin(rmse_values)].set_color('red')
        
        # Error Distribution
        errors = []
        labels = []
        for r in sklearn_results[:5]:
            if r['success'] and 'y_test' in r and 'y_pred' in r:
                error = np.abs(np.array(r['y_test']) - np.array(r['y_pred']))
                errors.append(error)
                labels.append(r['name'])
        
        if errors:
            ax[1].boxplot(errors, labels=labels)
            ax[1].set_ylabel('Absolute Error')
            ax[1].set_title('Error Distribution - Top Models')
            plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Figure 3: LSTM Training History
    fig3, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig3.suptitle('LSTM Training History', fontsize=16)
    
    ax.plot(lstm_results['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(lstm_results['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Figure 4: Feature Distributions
    fig4, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle('Feature Distributions', fontsize=16)
    
    ax[0].hist(df_raw['PL'], bins=30, alpha=0.7, label='PL')
    ax[0].set_xlabel('Path Loss (PL)')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Path Loss Distribution')
    ax[0].grid(True, alpha=0.3)
    
    ax[1].hist(df_raw['RMS'], bins=30, alpha=0.7, label='RMS', color='orange')
    ax[1].set_xlabel('RMS Delay Spread')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('RMS Delay Spread Distribution')
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def perform_statistical_analysis(sklearn_results, lstm_results):
    """Perform statistical analysis of results"""
    
    if not sklearn_results:
        return
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Extract RMSE values
    rmse_values = [r['metrics']['rmse'] for r in sklearn_results if r['success']]
    
    # Basic statistics
    print(f"\nModel Performance Statistics:")
    print(f"  Number of models: {len(rmse_values)}")
    print(f"  Mean RMSE: {np.mean(rmse_values):.4f}")
    print(f"  Std RMSE: {np.std(rmse_values):.4f}")
    print(f"  Min RMSE: {np.min(rmse_values):.4f}")
    print(f"  Max RMSE: {np.max(rmse_values):.4f}")
    print(f"  LSTM RMSE: {lstm_results['rmse']:.4f}")
    
    # Model type analysis
    model_types = {
        'Linear': ['linear', 'ridge', 'lasso', 'elastic', 'poly'],
        'SVM': ['svr']
    }
    
    print(f"\nPerformance by Model Type:")
    for type_name, keywords in model_types.items():
        type_rmses = [r['metrics']['rmse'] for r in sklearn_results 
                     if r['success'] and any(k in r['name'].lower() for k in keywords)]
        if type_rmses:
            print(f"  {type_name}: Mean RMSE = {np.mean(type_rmses):.4f} (n={len(type_rmses)})")

def save_analysis_results(sklearn_results, lstm_results, comparison_df):
    """Save analysis results to files"""
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Save detailed results
    results_summary = {
        'sklearn_results': [
            {
                'name': r['name'],
                'rmse': r['metrics']['rmse'],
                'mae': r['metrics']['mae'],
                'r2': r['metrics']['r2']
            }
            for r in sklearn_results if r['success']
        ],
        'lstm_results': {
            'rmse': lstm_results['rmse'],
            'final_train_loss': lstm_results['train_loss'][-1],
            'final_val_loss': lstm_results['val_loss'][-1]
        }
    }
    
    # Save to CSV
    if comparison_df is not None:
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        print(f"\nResults saved to results/model_comparison.csv")
    
    # Save detailed report
    with open('results/analysis_report.txt', 'w') as f:
        f.write("POSITION ESTIMATION ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("Top Models (Linear and SVR only):\n")
        for i, r in enumerate(sklearn_results[:10]):
            if r['success']:
                f.write(f"{i+1}. {r['name']}: RMSE={r['metrics']['rmse']:.4f}\n")
        
        f.write(f"\nLSTM Performance: RMSE={lstm_results['rmse']:.4f}\n")


if __name__ == "__main__":
    run_analysis()
    
    print("\nAnalysis complete.")