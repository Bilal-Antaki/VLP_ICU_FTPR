import numpy as np
import matplotlib.pyplot as plt
from src.training.train_sklearn import train_all_sklearn_models
from src.training.train_dl import train_lstm_on_all
from src.data.loader import load_cir_data, extract_features_and_target
from sklearn.model_selection import train_test_split

def run_comprehensive_comparison():
    """Run comprehensive comparison with proper size alignment"""
    processed_dir = "data/processed"
    
    print("=== COMPREHENSIVE MODEL COMPARISON ===\n")
    
    # Train sklearn models
    print("1. Training sklearn models...")
    sklearn_results, sklearn_df = train_all_sklearn_models(processed_dir)
    
    # Train LSTM
    print("\n2. Training LSTM model...")
    lstm_results = train_lstm_on_all(processed_dir)
    
    if sklearn_results and lstm_results:
        print("\n3. Creating comprehensive plots...")
        create_comprehensive_plots(sklearn_results, sklearn_df, lstm_results)
    else:
        print("Error: Could not complete training for all models")

def create_comprehensive_plots(sklearn_results, sklearn_df, lstm_results):
    """Create comprehensive plots with proper size alignment"""
    
    # Get the best sklearn model
    best_sklearn = min(sklearn_results, key=lambda x: x['rmse'])
    
    # Get LSTM results
    r_actual_lstm = np.array(lstm_results['r_actual'])
    r_pred_lstm = np.array(lstm_results['r_pred'])
    train_loss = lstm_results['train_loss']
    val_loss = lstm_results['val_loss']
    
    # Get sklearn predictions (these are on test set)
    sklearn_pred = np.array(best_sklearn['y_pred'])
    
    # Load original data to get proper alignment
    df = load_cir_data("data/processed", filter_keyword="FCPR-D1")
    X, y = extract_features_and_target(df)
    
    # Get the same test split used in sklearn training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sklearn_actual = np.array(y_test)
    
    print(f"Data sizes:")
    print(f"  Original dataset: {len(df)}")
    print(f"  LSTM sequences: {len(r_actual_lstm)}")
    print(f"  Sklearn test set: {len(sklearn_actual)}")
    print(f"  Best sklearn model: {best_sklearn['name']} (RMSE: {best_sklearn['rmse']:.4f})")
    print(f"  LSTM RMSE: {lstm_results['rmse']:.4f}")
    
    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comprehensive Model Comparison', fontsize=16)
    
    # Plot 1: RMSE Comparison Bar Chart
    model_names = [r['name'] for r in sklearn_results] + ['LSTM']
    rmse_values = [r['rmse'] for r in sklearn_results] + [lstm_results['rmse']]
    colors = ['skyblue'] * len(sklearn_results) + ['orange']
    
    bars = axes[0, 0].bar(range(len(model_names)), rmse_values, color=colors)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('RMSE Comparison Across All Models')
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, rmse_values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Sklearn Best Model - Actual vs Predicted
    axes[0, 1].scatter(sklearn_actual, sklearn_pred, alpha=0.6, color='blue')
    axes[0, 1].plot([sklearn_actual.min(), sklearn_actual.max()], 
                   [sklearn_actual.min(), sklearn_actual.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual r')
    axes[0, 1].set_ylabel('Predicted r')
    axes[0, 1].set_title(f'Best Sklearn Model: {best_sklearn["name"]}\nRMSE: {best_sklearn["rmse"]:.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: LSTM - Actual vs Predicted  
    axes[1, 0].scatter(r_actual_lstm, r_pred_lstm, alpha=0.6, color='orange')
    axes[1, 0].plot([r_actual_lstm.min(), r_actual_lstm.max()], 
                   [r_actual_lstm.min(), r_actual_lstm.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual r')
    axes[1, 0].set_ylabel('Predicted r')
    axes[1, 0].set_title(f'LSTM Model\nRMSE: {lstm_results["rmse"]:.4f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: LSTM Training History
    axes[1, 1].plot(train_loss, label='Training Loss', color='blue')
    axes[1, 1].plot(val_loss, label='Validation Loss', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE Loss')
    axes[1, 1].set_title('LSTM Training History')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    print(f"Best Sklearn Model: {best_sklearn['name']} - RMSE: {best_sklearn['rmse']:.4f}")
    print(f"LSTM Model: RMSE: {lstm_results['rmse']:.4f}")
    
    if lstm_results['rmse'] < best_sklearn['rmse']:
        improvement = ((best_sklearn['rmse'] - lstm_results['rmse']) / best_sklearn['rmse']) * 100
        print(f"\n🎉 LSTM outperforms best sklearn model by {improvement:.1f}%")
    else:
        difference = ((lstm_results['rmse'] - best_sklearn['rmse']) / best_sklearn['rmse']) * 100
        print(f"\n📊 Best sklearn model outperforms LSTM by {difference:.1f}%")

def run_simple_comparison():
    """Simple comparison without comprehensive plots"""
    processed_dir = "data/processed"
    
    print("=== SIMPLE MODEL COMPARISON ===\n")
    
    # Train sklearn models
    sklearn_results, sklearn_df = train_all_sklearn_models(processed_dir)
    
    # Train LSTM
    lstm_results = train_lstm_on_all(processed_dir)
    
    if sklearn_results and lstm_results:
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)
        
        best_sklearn = min(sklearn_results, key=lambda x: x['rmse'])
        print(f"Best Sklearn: {best_sklearn['name']} - RMSE: {best_sklearn['rmse']:.4f}")
        print(f"LSTM: RMSE: {lstm_results['rmse']:.4f}")
        
        if lstm_results['rmse'] < best_sklearn['rmse']:
            improvement = ((best_sklearn['rmse'] - lstm_results['rmse']) / best_sklearn['rmse']) * 100
            print(f"LSTM is better by {improvement:.1f}%")
        else:
            difference = ((lstm_results['rmse'] - best_sklearn['rmse']) / best_sklearn['rmse']) * 100
            print(f"Sklearn is better by {difference:.1f}%")

if __name__ == "__main__":
    # Set this to control which mode to run
    comprehensive_mode = False
    
    if comprehensive_mode:
        run_comprehensive_comparison()
    else:
        run_simple_comparison()