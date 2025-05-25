import pandas as pd
import matplotlib.pyplot as plt
from src.models.lstm_2 import train_sequence_lstm

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('data/processed/FCPR-D1_CIR.csv')
    
    # Train the model
    results = train_sequence_lstm(df, train_simulations=16, epochs=200)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.yscale('log')
    plt.title('Training History')
    plt.show()
    
    # Plot predictions for each test simulation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(4):
        axes[i].plot(range(10), results['actuals'][i], 'o-', label='Actual', markersize=8)
        axes[i].plot(range(10), results['predictions'][i], 's-', label='Predicted', markersize=6)
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Distance (r)')
        axes[i].set_title(f'Simulation {17+i} - RMSE: {results["sim_rmses"][i]:.2f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()