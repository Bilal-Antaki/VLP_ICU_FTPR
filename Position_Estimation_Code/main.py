from src.training.train_sklearn import train_linear_on_all, train_all_sklearn_models
from src.training.train_dl import train_lstm_on_all
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROCESSED_DATA_DIR = r"C:\Dev\Python\Position_Estimation_Code\data\processed"

def run_comprehensive_comparison():
    """Run comprehensive model comparison"""
    print("=" * 80)
    print("COMPREHENSIVE POSITION ESTIMATION COMPARISON")
    print("=" * 80)
    
    # 1. Train and compare all sklearn models
    print("\n1. Training and Comparing All Sklearn Models...")
    sklearn_results, sklearn_df = train_all_sklearn_models(PROCESSED_DATA_DIR)
    
    if sklearn_results:
        best_sklearn = min(sklearn_results, key=lambda x: x['rmse'])
        print(f"\nBest Sklearn Model: {best_sklearn['name']} (RMSE: {best_sklearn['rmse']:.4f})")
    
    # 2. Train LSTM
    print("\n2. Training LSTM...")
    r_actual_lstm, r_pred_lstm, train_loss, val_loss = train_lstm_on_all(PROCESSED_DATA_DIR)
    
    # 3. Create comprehensive plots
    if sklearn_results:
        create_comprehensive_plots(sklearn_results, sklearn_df, r_actual_lstm, r_pred_lstm, 
                                 train_loss, val_loss)
    
    return sklearn_results, r_actual_lstm, r_pred_lstm

def run_simple_comparison():
    """Run simple linear vs LSTM comparison (original behavior)"""
    print("=" * 60)
    print("POSITION ESTIMATION COMPARISON")
    print("=" * 60)
    
    # Train Linear
    print("\n1. Training Linear Regression...")
    r_actual, r_pred_linear = train_linear_on_all(PROCESSED_DATA_DIR)

    # Train LSTM
    print("\n2. Training LSTM...")
    r_actual_lstm, r_pred_lstm, train_loss, val_loss = train_lstm_on_all(PROCESSED_DATA_DIR)

    # Align lengths
    min_len = min(len(r_actual), len(r_actual_lstm))
    r_actual = r_actual[-min_len:]
    r_pred_linear = r_pred_linear[-min_len:]
    r_pred_lstm = r_pred_lstm[-min_len:]

    print(f"\n3. Comparing results (showing {min_len} samples)...")
    
    # Calculate metrics
    def calculate_metrics(actual, predicted, model_name):
        rmse = np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))
        mae = np.mean(np.abs(np.array(actual) - np.array(predicted)))
        r2 = 1 - (np.sum((np.array(actual) - np.array(predicted)) ** 2) / 
                  np.sum((np.array(actual) - np.mean(actual)) ** 2))
        
        print(f"\n{model_name} Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  Pred Range: [{min(predicted):.2f}, {max(predicted):.2f}]")
        print(f"  Pred Std:   {np.std(predicted):.4f}")
        
        return rmse, mae, r2

    # Calculate metrics for both models
    linear_rmse, linear_mae, linear_r2 = calculate_metrics(r_actual, r_pred_linear, "Linear Regression")
    lstm_rmse, lstm_mae, lstm_r2 = calculate_metrics(r_actual_lstm, r_pred_lstm, "LSTM")
    
    print(f"\nActual Target Statistics:")
    print(f"  Range: [{min(r_actual):.2f}, {max(r_actual):.2f}]")
    print(f"  Mean:  {np.mean(r_actual):.2f}")
    print(f"  Std:   {np.std(r_actual):.4f}")
    
    # Create comparison DataFrame
    df_compare = pd.DataFrame({
        "r_actual": r_actual,
        "r_pred_linear": r_pred_linear,
        "r_pred_lstm": r_pred_lstm,
        "linear_error": np.abs(np.array(r_actual) - np.array(r_pred_linear)),
        "lstm_error": np.abs(np.array(r_actual_lstm) - np.array(r_pred_lstm))
    }).round(3)
    
    print(f"\nSample Predictions:")
    print(df_compare.head(15))
    
    # Create plots
    create_simple_plots(r_actual, r_pred_linear, r_actual_lstm, r_pred_lstm, 
                       train_loss, val_loss, linear_r2, lstm_r2, df_compare)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Linear Regression - RMSE: {linear_rmse:.4f}, R²: {linear_r2:.4f}")
    print(f"LSTM             - RMSE: {lstm_rmse:.4f}, R²: {lstm_r2:.4f}")
    
    if lstm_rmse < linear_rmse:
        improvement = ((linear_rmse - lstm_rmse) / linear_rmse) * 100
        print(f"LSTM shows {improvement:.2f}% improvement over Linear Regression")
    else:
        degradation = ((lstm_rmse - linear_rmse) / linear_rmse) * 100
        print(f"Linear Regression performs {degradation:.2f}% better than LSTM")
    
    print("=" * 60)

def create_simple_plots(r_actual, r_pred_linear, r_actual_lstm, r_pred_lstm, 
                       train_loss, val_loss, linear_r2, lstm_r2, df_compare):
    """Create simple comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: LSTM Training Loss
    axes[0, 0].plot(train_loss, label="Train Loss", alpha=0.8)
    axes[0, 0].plot(val_loss, label="Val Loss", alpha=0.8)
    axes[0, 0].set_title("LSTM Training History")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Predictions vs Actual
    axes[0, 1].scatter(r_actual, r_pred_linear, alpha=0.6, label=f'Linear (R²={linear_r2:.3f})', s=20)
    axes[0, 1].scatter(r_actual_lstm, r_pred_lstm, alpha=0.6, label=f'LSTM (R²={lstm_r2:.3f})', s=20)
    
    # Perfect prediction line
    min_val = min(min(r_actual), min(r_actual_lstm))
    max_val = max(max(r_actual), max(r_actual_lstm))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect')
    
    axes[0, 1].set_xlabel("Actual r")
    axes[0, 1].set_ylabel("Predicted r")
    axes[0, 1].set_title("Predictions vs Actual")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error Distribution
    axes[1, 0].hist(df_compare['linear_error'], bins=30, alpha=0.7, label='Linear', density=True)
    axes[1, 0].hist(df_compare['lstm_error'], bins=30, alpha=0.7, label='LSTM', density=True)
    axes[1, 0].set_xlabel("Absolute Error")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Error Distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Prediction Time Series
    sample_indices = range(min(100, len(r_actual)))
    axes[1, 1].plot(sample_indices, [r_actual[i] for i in sample_indices], 'o-', label='Actual', alpha=0.8, markersize=4)
    axes[1, 1].plot(sample_indices, [r_pred_linear[i] for i in sample_indices], 's-', label='Linear', alpha=0.8, markersize=3)
    axes[1, 1].plot(sample_indices, [r_pred_lstm[i] for i in sample_indices], '^-', label='LSTM', alpha=0.8, markersize=3)
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("r value")
    axes[1, 1].set_title("Prediction Comparison (First 100 samples)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_comprehensive_plots(sklearn_results, sklearn_df, r_actual_lstm, r_pred_lstm, 
                             train_loss, val_loss):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: LSTM Training Loss
    axes[0, 0].plot(train_loss, label="Train Loss", alpha=0.8)
    axes[0, 0].plot(val_loss, label="Val Loss", alpha=0.8)
    axes[0, 0].set_title("LSTM Training History")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Sklearn Model Comparison
    model_names = [r['name'] for r in sklearn_results]
    rmse_values = [r['rmse'] for r in sklearn_results]
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    bars = axes[0, 1].bar(range(len(model_names)), rmse_values, color=colors)
    axes[0, 1].set_xlabel("Models")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].set_title("Sklearn Model Performance")
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rmse in zip(bars, rmse_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{rmse:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: R² Comparison
    r2_values = [r['r2'] for r in sklearn_results]
    bars = axes[0, 2].bar(range(len(model_names)), r2_values, color=colors)
    axes[0, 2].set_xlabel("Models")
    axes[0, 2].set_ylabel("R²")
    axes[0, 2].set_title("Model R² Scores")
    axes[0, 2].set_xticks(range(len(model_names)))
    axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, r2 in zip(bars, r2_values):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{r2:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Best Models vs LSTM Predictions
    best_sklearn = min(sklearn_results, key=lambda x: x['rmse'])
    axes[1, 0].scatter(r_actual_lstm, best_sklearn['y_pred'][:len(r_actual_lstm)], 
                      alpha=0.6, label=f'Best Sklearn: {best_sklearn["name"]}', s=20)
    axes[1, 0].scatter(r_actual_lstm, r_pred_lstm, alpha=0.6, label='LSTM', s=20)
    
    min_val = min(min(r_actual_lstm), min(r_pred_lstm))
    max_val = max(max(r_actual_lstm), max(r_pred_lstm))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect')
    
    axes[1, 0].set_xlabel("Actual r")
    axes[1, 0].set_ylabel("Predicted r")
    axes[1, 0].set_title("Best Models vs Actual")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Model Complexity vs Performance
    model_complexity = [1, 2, 2, 3, 3, 4, 5, 5, 6]  # Rough complexity scores
    if len(model_complexity) == len(sklearn_results):
        axes[1, 1].scatter(model_complexity, rmse_values, s=100, alpha=0.7, c=colors)
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (model_complexity[i], rmse_values[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel("Model Complexity")
        axes[1, 1].set_ylabel("RMSE")
        axes[1, 1].set_title("Complexity vs Performance")
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Summary Table (as text)
    axes[1, 2].axis('off')
    summary_text = "MODEL SUMMARY\n" + "="*40 + "\n"
    summary_text += f"Best Sklearn: {best_sklearn['name']}\n"
    summary_text += f"  RMSE: {best_sklearn['rmse']:.4f}\n"
    summary_text += f"  R²: {best_sklearn['r2']:.4f}\n\n"
    
    lstm_rmse = np.sqrt(np.mean((np.array(r_actual_lstm) - np.array(r_pred_lstm)) ** 2))
    lstm_r2 = 1 - (np.sum((np.array(r_actual_lstm) - np.array(r_pred_lstm)) ** 2) / 
                   np.sum((np.array(r_actual_lstm) - np.mean(r_actual_lstm)) ** 2))
    
    summary_text += f"LSTM:\n"
    summary_text += f"  RMSE: {lstm_rmse:.4f}\n"
    summary_text += f"  R²: {lstm_r2:.4f}\n\n"
    
    if lstm_rmse < best_sklearn['rmse']:
        improvement = ((best_sklearn['rmse'] - lstm_rmse) / best_sklearn['rmse']) * 100
        summary_text += f"LSTM is {improvement:.1f}% better"
    else:
        degradation = ((lstm_rmse - best_sklearn['rmse']) / best_sklearn['rmse']) * 100
        summary_text += f"Sklearn is {degradation:.1f}% better"
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                   verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Choose your comparison mode
    comprehensive_mode = True  # Set to False for simple linear vs LSTM comparison
    
    if comprehensive_mode:
        run_comprehensive_comparison()
    else:
        run_simple_comparison()