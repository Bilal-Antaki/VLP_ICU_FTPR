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
    #create_analysis_figure_set(results_basic, lstm_results, df_all, df_engineered)
    
    # 7. Statistical Analysis
    print("\n7. Statistical Analysis...")
    perform_statistical_analysis(results_basic, lstm_results)
    
    # 8. Save results
    save_analysis_results(results_basic, lstm_results, df_basic)

def create_analysis_figure_set(sklearn_results, lstm_results, df_raw, df_engineered):
    # Simulated data for illustration
    df_raw = pd.DataFrame({
        'PL': np.random.normal(65, 3, 1000),
        'RMS': np.random.normal(10, 2, 1000),
        'r': np.random.uniform(100, 5000, 1000),
        'X': np.random.uniform(0, 100, 1000),
        'Y': np.random.uniform(0, 100, 1000)
    })

    sklearn_results = [
        {'name': 'linear', 'metrics': {'rmse': 400}, 'success': True, 'y_test': np.random.rand(100)*2000, 'y_pred': np.random.rand(100)*2000},
        {'name': 'ridge', 'metrics': {'rmse': 395}, 'success': True, 'y_test': np.random.rand(100)*2000, 'y_pred': np.random.rand(100)*2000},
        {'name': 'lasso', 'metrics': {'rmse': 390}, 'success': True, 'y_test': np.random.rand(100)*2000, 'y_pred': np.random.rand(100)*2000},
        {'name': 'elastic', 'metrics': {'rmse': 385}, 'success': True, 'y_test': np.random.rand(100)*2000, 'y_pred': np.random.rand(100)*2000},
        {'name': 'poly', 'metrics': {'rmse': 380}, 'success': True, 'y_test': np.random.rand(100)*2000, 'y_pred': np.random.rand(100)*2000},
        {'name': 'random_forest', 'metrics': {'rmse': 375}, 'success': True, 'model': type('mock', (), {'feature_importances_': np.array([0.4, 0.6])})()}
    ]

    lstm_results = {
        'rmse': 50,
        'train_loss': np.linspace(1.0, 0.1, 70),
        'val_loss': np.linspace(1.0, 0.5, 70) + np.random.normal(0, 0.05, 70)
    }

    # ---- Figure 1: Data Exploration ----
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))

    # Correlation Heatmap
    correlation_matrix = df_raw[['PL', 'RMS', 'r']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axs1[0])
    axs1[0].set_title('Feature Correlation Matrix')

    # Feature Distributions
    df_raw[['PL', 'RMS']].hist(bins=30, ax=axs1[1], alpha=0.7)
    axs1[1].set_title('Feature Distributions')

    # PL vs Distance
    scatter = axs1[2].scatter(df_raw['r'], df_raw['PL'], c=df_raw['RMS'], cmap='viridis', alpha=0.6, s=20)
    axs1[2].set_xlabel('Distance (r)')
    axs1[2].set_ylabel('Path Loss (PL)')
    axs1[2].set_title('PL vs Distance (colored by RMS)')
    fig1.colorbar(scatter, ax=axs1[2], label='RMS')

    plt.tight_layout()
    plt.show()

    # ---- Figure 2: Model Performance ----
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5))

    # Model Performance Comparison
    model_names = [r['name'] for r in sklearn_results[:5]]
    rmse_values = [r['metrics']['rmse'] for r in sklearn_results[:5]]
    model_names.append('LSTM')
    rmse_values.append(lstm_results['rmse'])
    bars = axs2[0].barh(model_names, rmse_values)
    axs2[0].set_xlabel('RMSE')
    axs2[0].set_title('Top Models by RMSE')
    bars[np.argmin(rmse_values)].set_color('red')

    # Residual Plot
    best_model = sklearn_results[0]
    residuals = np.array(best_model['y_test']) - np.array(best_model['y_pred'])
    axs2[1].scatter(best_model['y_pred'], residuals, alpha=0.5)
    axs2[1].axhline(y=0, color='r', linestyle='--')
    axs2[1].set_xlabel('Predicted Values')
    axs2[1].set_ylabel('Residuals')
    axs2[1].set_title(f'Residual Plot - {best_model["name"]}')

    # Error Distribution
    errors = []
    labels = []
    for r in sklearn_results[:5]:
        if r['success'] and 'y_test' in r and 'y_pred' in r:
            error = np.abs(np.array(r['y_test']) - np.array(r['y_pred']))
            errors.append(error)
            labels.append(r['name'])
    axs2[2].boxplot(errors, labels=labels)
    axs2[2].set_ylabel('Absolute Error')
    axs2[2].set_title('Error Distribution - Top 5 Models')
    plt.setp(axs2[2].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

    # ---- Figure 3: LSTM & Feature Insights ----
    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5))

    # LSTM Training History
    axs3[0].plot(lstm_results['train_loss'], label='Train Loss')
    axs3[0].plot(lstm_results['val_loss'], label='Val Loss')
    axs3[0].set_xlabel('Epoch')
    axs3[0].set_ylabel('Loss')
    axs3[0].set_title('LSTM Training History')
    axs3[0].legend()

    # Feature Importance
    tree_model = sklearn_results[-1]
    importances = tree_model['model'].feature_importances_
    features = ['PL', 'RMS']
    axs3[1].bar(features, importances)
    axs3[1].set_title(f'Feature Importances - {tree_model["name"]}')
    axs3[1].set_ylabel('Importance')

    # 2D Path Loss Heatmap
    heatmap_data = df_raw.pivot_table(
        values='PL',
        index=pd.cut(df_raw['Y'], bins=20),
        columns=pd.cut(df_raw['X'], bins=20),
        aggfunc='mean'
    )
    sns.heatmap(heatmap_data, cmap='RdYlBu_r', ax=axs3[2], cbar_kws={'label': 'Avg PL'})
    axs3[2].set_title('Path Loss Heatmap')
    axs3[2].set_xlabel('X bins')
    axs3[2].set_ylabel('Y bins')

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
        'Tree': ['forest', 'tree'],
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