# main_enhanced.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.training.train_enhanced import train_all_models_enhanced, automated_model_selection
from src.training.train_dl import train_lstm_on_all
from src.data.loader import load_cir_data, extract_features_and_target
from src.data.feature_engineering import create_engineered_features, select_features
from src.evaluation.metrics import print_metrics_report, calculate_all_metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def run_comprehensive_analysis():
    """Run comprehensive analysis with all models and feature engineering"""
    processed_dir = "data/processed"
    
    print("=" * 80)
    print("COMPREHENSIVE POSITION ESTIMATION ANALYSIS")
    print("=" * 80)
    
    # 1. Load and explore data
    print("\n1. Loading and exploring data...")
    df_list = []
    
    # Load all available datasets
    for keyword in ['FCPR-D1', 'FCPR-D2', 'FCPR-D3', 'ICU-D1', 'ICU-D2', 'ICU-D3']:
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
    print("\n2. Feature Engineering...")
    df_engineered = create_engineered_features(df_all, include_coordinates=True, include_categorical=False)
    
    # Select features
    feature_cols = [col for col in df_engineered.columns 
                   if col not in ['r', 'X', 'Y', 'source_file']]
    
    X = df_engineered[feature_cols]
    y = df_engineered['r']
    
    # Select best features
    selected_features = select_features(X, y, method='correlation', threshold=0.3)
    print(f"  Selected {len(selected_features)} features from {len(feature_cols)} total")
    print(f"  Top features: {selected_features[:10]}")
    
    X_selected = X[selected_features]
    
    # 3. Train models with basic features
    print("\n3. Training models with BASIC features (PL, RMS only)...")
    results_basic, df_basic, _ = train_all_models_enhanced(
        processed_dir, 
        include_slow_models=True
    )
    
    # 4. Train models with engineered features
    print("\n4. Training models with ENGINEERED features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Convert to temporary CSV for compatibility
    temp_df = pd.concat([X_selected, y], axis=1)
    temp_df['source_file'] = 'engineered'
    temp_df.to_csv('data/processed/temp_engineered_CIR.csv', index=False)
    
    # Note: This would need modification of train_all_models_enhanced to accept custom features
    # For now, we'll show the comparison concept
    
    # 5. Deep Learning Comparison
    print("\n5. Training LSTM model...")
    lstm_results = train_lstm_on_all(processed_dir)
    
    # 6. Create comprehensive visualization
    create_comprehensive_analysis_plots(results_basic, lstm_results, df_all, df_engineered)
    
    # 7. Statistical Analysis
    print("\n7. Statistical Analysis...")
    perform_statistical_analysis(results_basic, lstm_results)
    
    # 8. Save results
    save_analysis_results(results_basic, lstm_results, df_basic)

def create_comprehensive_analysis_plots(sklearn_results, lstm_results, df_raw, df_engineered):
    """Create comprehensive analysis plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Feature Correlation Heatmap
    ax1 = plt.subplot(3, 3, 1)
    correlation_matrix = df_raw[['PL', 'RMS', 'r']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title('Feature Correlation Matrix')
    
    # 2. Feature Distributions
    ax2 = plt.subplot(3, 3, 2)
    df_raw[['PL', 'RMS']].hist(bins=30, ax=ax2, alpha=0.7)
    ax2.set_title('Feature Distributions')
    
    # 3. PL vs Distance scatter
    ax3 = plt.subplot(3, 3, 3)
    scatter = ax3.scatter(df_raw['r'], df_raw['PL'], c=df_raw['RMS'], 
                         cmap='viridis', alpha=0.6, s=20)
    ax3.set_xlabel('Distance (r)')
    ax3.set_ylabel('Path Loss (PL)')
    ax3.set_title('PL vs Distance (colored by RMS)')
    plt.colorbar(scatter, ax=ax3, label='RMS')
    
    # 4. Model Performance Comparison
    ax4 = plt.subplot(3, 3, 4)
    if sklearn_results:
        model_names = [r['name'] for r in sklearn_results[:10]]  # Top 10
        rmse_values = [r['metrics']['rmse'] for r in sklearn_results[:10]]
        
        # Add LSTM
        model_names.append('LSTM')
        rmse_values.append(lstm_results['rmse'])
        
        bars = ax4.barh(model_names, rmse_values)
        ax4.set_xlabel('RMSE')
        ax4.set_title('Top 10 Models by RMSE')
        
        # Color best model
        min_idx = np.argmin(rmse_values)
        bars[min_idx].set_color('red')
    
    # 5. Residual Analysis
    ax5 = plt.subplot(3, 3, 5)
    if sklearn_results and sklearn_results[0]['success']:
        best_model = sklearn_results[0]
        if 'y_test' in best_model and 'y_pred' in best_model:
            residuals = np.array(best_model['y_test']) - np.array(best_model['y_pred'])
            ax5.scatter(best_model['y_pred'], residuals, alpha=0.5)
            ax5.axhline(y=0, color='r', linestyle='--')
            ax5.set_xlabel('Predicted Values')
            ax5.set_ylabel('Residuals')
            ax5.set_title(f'Residual Plot - {best_model["name"]}')
        else:
            ax5.text(0.5, 0.5, 'No residual data available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Residual Plot - Not Available')
    # 6. LSTM Training History
    ax6 = plt.subplot(3, 3, 6)
    if lstm_results:
        ax6.plot(lstm_results['train_loss'], label='Train Loss')
        ax6.plot(lstm_results['val_loss'], label='Val Loss')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss')
        ax6.set_title('LSTM Training History')
        ax6.legend()
    
    # 7. Feature Importance (for tree-based models)
    ax7 = plt.subplot(3, 3, 7)
    # Find a tree-based model
    tree_model = None
    for r in sklearn_results:
        if 'forest' in r['name'].lower() or 'boost' in r['name'].lower():
            tree_model = r
            break
    
    if tree_model and hasattr(tree_model['model'], 'feature_importances_'):
        importances = tree_model['model'].feature_importances_
        features = ['PL', 'RMS']
        ax7.bar(features, importances)
        ax7.set_title(f'Feature Importances - {tree_model["name"]}')
        ax7.set_ylabel('Importance')
    
    # 8. Error Distribution
    ax8 = plt.subplot(3, 3, 8)
    if sklearn_results:
        errors = []
        labels = []
        for r in sklearn_results[:5]:  # Top 5 models
            if r['success'] and 'y_test' in r and 'y_pred' in r:
                error = np.abs(np.array(r['y_test']) - np.array(r['y_pred']))
                errors.append(error)
                labels.append(r['name'])
        
        if errors:
            ax8.boxplot(errors, labels=labels)
            ax8.set_ylabel('Absolute Error')
            ax8.set_title('Error Distribution - Top 5 Models')
            plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax8.text(0.5, 0.5, 'No error data available', 
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Error Distribution - Not Available')
    
    # 9. 2D Position Heatmap
    ax9 = plt.subplot(3, 3, 9)
    if 'X' in df_raw.columns and 'Y' in df_raw.columns:
        heatmap_data = df_raw.pivot_table(
            values='PL', 
            index=pd.cut(df_raw['Y'], bins=20), 
            columns=pd.cut(df_raw['X'], bins=20),
            aggfunc='mean'
        )
        sns.heatmap(heatmap_data, cmap='RdYlBu_r', ax=ax9, cbar_kws={'label': 'Avg PL'})
        ax9.set_title('Path Loss Heatmap')
        ax9.set_xlabel('X bins')
        ax9.set_ylabel('Y bins')
    
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
    
    # Performance categories
    excellent = sum(1 for r in rmse_values if r < 50)
    good = sum(1 for r in rmse_values if 50 <= r < 100)
    fair = sum(1 for r in rmse_values if 100 <= r < 150)
    poor = sum(1 for r in rmse_values if r >= 150)
    
    print(f"\nPerformance Distribution:")
    print(f"  Excellent (<50): {excellent}")
    print(f"  Good (50-100): {good}")
    print(f"  Fair (100-150): {fair}")
    print(f"  Poor (>150): {poor}")
    
    # Model type analysis
    model_types = {
        'Linear': ['linear'],
        'SVM': ['svr'],
        # 'Tree': ['forest', 'tree'],
        #'Neural': ['mlp'],
        #'Neighbors': ['knn']
    }
    
    print(f"\nPerformance by Model Type:")
    for type_name, keywords in model_types.items():
        type_rmses = [r['metrics']['rmse'] for r in sklearn_results 
                     if r['success'] and any(k in r['name'].lower() for k in keywords)]
        if type_rmses:
            print(f"  {type_name}: Mean RMSE = {np.mean(type_rmses):.4f} (n={len(type_rmses)})")

def save_analysis_results(sklearn_results, lstm_results, comparison_df):
    """Save analysis results to files"""
    
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
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Save detailed report
    with open('results/analysis_report.txt', 'w') as f:
        f.write("POSITION ESTIMATION ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("Top 10 Models:\n")
        for i, r in enumerate(sklearn_results[:10]):
            if r['success']:
                f.write(f"{i+1}. {r['name']}: RMSE={r['metrics']['rmse']:.4f}\n")
        
        f.write(f"\nLSTM Performance: RMSE={lstm_results['rmse']:.4f}\n")

if __name__ == "__main__":
    # Run comprehensive analysis
    run_comprehensive_analysis()
    
    # Or run automated model selection
    #best_model = automated_model_selection("data/processed")
    
    print("\nAnalysis complete!")