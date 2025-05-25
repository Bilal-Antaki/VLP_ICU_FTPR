# main.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.training.train_sklearn import train_all_models_enhanced
from src.training.train_dl import train_lstm_on_all
from src.data.loader import load_cir_data
from src.data.feature_engineering import create_engineered_features, select_features
from src.config import DATA_CONFIG, ANALYSIS_CONFIG, TRAINING_OPTIONS
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

def run_analysis():
    """Run analysis with Linear, SVR, and LSTM models"""
    print("=" * 80)
    print(" " * 20 + "COMPREHENSIVE POSITION ESTIMATION ANALYSIS")
    print("=" * 80)
    
    # 1. Load and explore data
    print("\n1. Loading and exploring data...")
    df_list = []
    
    # Load all available datasets
    for keyword in DATA_CONFIG['datasets']:
        try:
            df_temp = load_cir_data(DATA_CONFIG['processed_dir'], filter_keyword=keyword)
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
    print("\n2. Feature Engineering...")
    df_engineered = create_engineered_features(df_all, include_categorical=True)
    
    # Select features - exclude any coordinate-based features
    feature_cols = [col for col in df_engineered.columns 
                   if col not in ANALYSIS_CONFIG['feature_selection']['excluded_features']]
    
    X = df_engineered[feature_cols]
    y = df_engineered[DATA_CONFIG['target_column']]

    # Select best features
    selected_features = select_features(
        X, y, 
        method='correlation', 
        threshold=ANALYSIS_CONFIG['feature_selection']['correlation_threshold']
    )
    print(f"  Selected {len(selected_features)} features from {len(feature_cols)} total")
    print(f"  Top features: {selected_features[:10]}")
    
    # 3. Train all models and collect results
    print("\n3. Training all models...")
    all_model_results = []
    
    # Train traditional ML models
    sklearn_results = train_all_models_enhanced(
        DATA_CONFIG['processed_dir'], 
        include_slow_models=TRAINING_OPTIONS['include_slow_models']
    )
    
    # Add successful sklearn results to all_model_results
    for result in sklearn_results:
        if result['success']:
            all_model_results.append({
                'name': result['name'],
                'type': 'sklearn',
                'metrics': result['metrics'],
                'predictions': {
                    'y_test': result.get('y_test', []),
                    'y_pred': result.get('y_pred', [])
                } if TRAINING_OPTIONS['save_predictions'] else None,
                'training_history': None
            })
    
    # Train and add LSTM results
    lstm_results = train_lstm_on_all(DATA_CONFIG['processed_dir'])
    all_model_results.append({
        'name': 'LSTM',
        'type': 'RNN',
        'metrics': {
            'rmse': lstm_results['rmse'],
            'mae': lstm_results.get('mae', None),
            'r2': lstm_results.get('r2', None)
        },
        'predictions': {
            'y_test': lstm_results.get('y_test', []),
            'y_pred': lstm_results.get('y_pred', [])
        } if TRAINING_OPTIONS['save_predictions'] else None,
        'training_history': {
            'train_loss': lstm_results['train_loss'],
            'val_loss': lstm_results['val_loss']
        } if TRAINING_OPTIONS['plot_training_history'] else None
    })
    
    # 4. Create visualization figures
    if TRAINING_OPTIONS['plot_training_history']:
        create_analysis_figures(all_model_results, df_all)
    
    # 5. Statistical Analysis
    print("\n4. Statistical Analysis...")
    perform_statistical_analysis(all_model_results)
    
    # 6. Save results
    if TRAINING_OPTIONS['save_predictions']:
        print("\n5. Saving Results...")
        save_analysis_results(all_model_results)
    
    print("\nAnalysis complete.")

def create_analysis_figures(model_results, df_raw):
    """Create separate figure windows for different visualizations"""
    
    # Figure 1: Data Exploration
    fig1, ax = plt.subplots(1, 2, figsize=ANALYSIS_CONFIG['visualization']['figure_sizes']['data_exploration'])
    fig1.suptitle('Data Exploration', fontsize=16)
    
    # Correlation Heatmap
    correlation_matrix = df_raw[['PL', 'RMS', 'r']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax[0])
    ax[0].set_title('Feature Correlation Matrix')
    
    # PL vs Distance
    scatter = ax[1].scatter(
        df_raw['r'], 
        df_raw['PL'], 
        c=df_raw['RMS'], 
        cmap='viridis', 
        alpha=ANALYSIS_CONFIG['visualization']['scatter_alpha'], 
        s=ANALYSIS_CONFIG['visualization']['scatter_size']
    )
    ax[1].set_xlabel('Distance (r)')
    ax[1].set_ylabel('Path Loss (PL)')
    ax[1].set_title('PL vs Distance (colored by RMS)')
    fig1.colorbar(scatter, ax=ax[1], label='RMS')
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Model Performance Comparison
    fig2, axes = plt.subplots(
        2, 1, 
        figsize=ANALYSIS_CONFIG['visualization']['figure_sizes']['model_comparison'], 
        height_ratios=ANALYSIS_CONFIG['visualization']['height_ratios']
    )
    fig2.suptitle('Model Performance Comparison', fontsize=16)
    
    # Sort models by RMSE for better visualization
    model_results.sort(key=lambda x: x['metrics']['rmse'])
    
    # Model Performance Comparison (RMSE)
    model_names = [r['name'] for r in model_results]
    rmse_values = [r['metrics']['rmse'] for r in model_results]
    
    bars = axes[0].barh(model_names, rmse_values)
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('Model Performance by RMSE')
    axes[0].grid(True, alpha=ANALYSIS_CONFIG['visualization']['grid_alpha'])
    
    # Color code bars by model type
    for i, result in enumerate(model_results):
        if result['type'] == 'RNN':
            bars[i].set_color('red')
        else:
            bars[i].set_color('blue')
    
    # Training History for models that have it
    axes[1].set_title('Training History')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=ANALYSIS_CONFIG['visualization']['grid_alpha'])
    axes[1].set_yscale('log')
    
    for result in model_results:
        if result['training_history']:
            if 'train_loss' in result['training_history']:
                axes[1].plot(
                    result['training_history']['train_loss'],
                    label=f"{result['name']} (Train)",
                    linewidth=2
                )
            if 'val_loss' in result['training_history']:
                axes[1].plot(
                    result['training_history']['val_loss'],
                    label=f"{result['name']} (Val)",
                    linewidth=2,
                    linestyle='--'
                )
    
    axes[1].legend()
    plt.tight_layout()
    plt.show()

def perform_statistical_analysis(model_results):
    """Perform statistical analysis of results"""
    print("\nSTATISTICAL ANALYSIS")
    print("="*60)
    
    # Overall statistics
    rmse_values = [r['metrics']['rmse'] for r in model_results]
    mae_values = [r['metrics'].get('mae', 0) for r in model_results]
    r2_values = [r['metrics'].get('r2', 0) for r in model_results]
    
    print("\nOverall Model Performance:")
    print(f"  Mean RMSE: {np.mean(rmse_values):.4f}")
    print(f"  Best RMSE: {np.min(rmse_values):.4f}")
    print(f"  Std RMSE: {np.std(rmse_values):.4f}")
    
    # Only print MAE and R2 statistics if available for all models
    if all(mae_values):
        print(f"  Mean MAE: {np.mean(mae_values):.4f}")
    if all(r2_values):
        print(f"  Mean R2: {np.mean(r2_values):.4f}")
    
    # List all models trained
    model_names = [r['name'] for r in model_results]
    print(f"\nModels trained: {', '.join(model_names)}")
    
    # Individual model results
    print("\nDetailed Model Results:")
    print("-" * 40)
    
    # Sort results by RMSE for better readability
    sorted_results = sorted(model_results, key=lambda x: x['metrics']['rmse'])
    best_rmse = min(rmse_values)
    
    for result in sorted_results:
        print(f"\n{result['name']}:")
        print(f"  RMSE: {result['metrics']['rmse']:.4f}")
        if result['metrics'].get('mae'):
            print(f"  MAE: {result['metrics']['mae']:.4f}")
        if result['metrics'].get('r2'):
            print(f"  R2: {result['metrics']['r2']:.4f}")
        
        # Show if this was the best model
        if result['metrics']['rmse'] == best_rmse:
            print("  → Best performing model")

def save_analysis_results(model_results):
    """Save analysis results to files"""
    
    # Create results directory if it doesn't exist
    os.makedirs(ANALYSIS_CONFIG['output']['results_dir'], exist_ok=True)
    
    # Save detailed report
    report_path = os.path.join(
        ANALYSIS_CONFIG['output']['results_dir'], 
        ANALYSIS_CONFIG['output']['report_file']
    )
    
    with open(report_path, 'w') as f:
        f.write("POSITION ESTIMATION ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Sort models by RMSE
        sorted_results = sorted(model_results, key=lambda x: x['metrics']['rmse'])
        
        f.write("Model Performance Summary:\n")
        f.write("-"*40 + "\n")
        for result in sorted_results:
            f.write(f"\nModel: {result['name']} ({result['type']})\n")
            f.write(f"  RMSE: {result['metrics']['rmse']:.4f}\n")
            if result['metrics'].get('mae'):
                f.write(f"  MAE: {result['metrics']['mae']:.4f}\n")
            if result['metrics'].get('r2'):
                f.write(f"  R2: {result['metrics']['r2']:.4f}\n")


if __name__ == "__main__":
    run_analysis()