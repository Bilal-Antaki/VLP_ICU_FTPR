import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.models.model_registry import get_model
from src.data.loader import load_cir_data
from src.data.preprocessing import scale_and_sequence
import numpy as np

def train_gru_on_all(processed_dir: str, model_variant: str = "gru", 
                     batch_size: int = 32, epochs: int = 300, 
                     lr: float = 0.01, seq_len: int = 10):
    """
    Train GRU model with specific optimizations for position estimation
    
    Args:
        processed_dir: Directory with processed data
        model_variant: "gru", "gru_attention", "gru_bidirectional", or "gru_residual"
        batch_size: Batch size for training
        epochs: Number of epochs
        lr: Learning rate
        seq_len: Sequence length for time series
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    
    # Load data
    df = load_cir_data(processed_dir, filter_keyword="FCPR-D1")
    print(f"Loaded {len(df)} data points")
    
    # Scale and create sequences
    X_seq, y_seq, x_scaler, y_scaler = scale_and_sequence(df, seq_len=seq_len)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Create data loaders with same batch sizes as LSTM
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=batch_size,  # Same as training batch size, like LSTM
        drop_last=False
    )
    
    # Create model
    model = get_model(model_variant, input_dim=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Using {model_variant} model on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer - use same as LSTM
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_loss_hist, val_loss_hist = [], []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d}: Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}")
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss > best_val_loss:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Generate predictions on full dataset
    model.eval()
    with torch.no_grad():
        full_preds_scaled = model(X_seq.to(device)).cpu().numpy()
        full_targets_scaled = y_seq.numpy()
    
    # Inverse transform
    full_preds = y_scaler.inverse_transform(full_preds_scaled.reshape(-1, 1)).flatten()
    full_targets = y_scaler.inverse_transform(full_targets_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(np.mean((full_targets - full_preds) ** 2))
    
    print(f"\nFinal Metrics for {model_variant}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return {
        'r_actual': full_targets.tolist(),
        'r_pred': full_preds.tolist(),
        'train_loss': train_loss_hist,
        'val_loss': val_loss_hist,
        'rmse': rmse,
        'model': model,
        'scalers': (x_scaler, y_scaler)
    }