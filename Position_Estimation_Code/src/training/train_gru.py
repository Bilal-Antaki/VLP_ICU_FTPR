import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.data.loader import load_cir_data
from src.data.preprocessing import scale_and_sequence
from src.config import DATA_CONFIG
import numpy as np
import pandas as pd
import random
import time
import math

class ImprovedGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3, output_activation=None):
        super(ImprovedGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Multi-layer GRU with dropout
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout layer after GRU
        self.dropout = nn.Dropout(dropout)
        
        # Multi-layer output head with residual connection
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc_out = nn.Linear(hidden_dim // 4, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.output_activation = output_activation
        
        # Skip connection
        self.skip_connection = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Multi-layer output head
        out1 = self.leaky_relu(self.bn1(self.fc1(last_hidden)))
        out1 = self.dropout(out1)
        
        out2 = self.leaky_relu(self.bn2(self.fc2(out1)))
        out2 = self.dropout(out2)
        
        main_output = self.fc_out(out2)
        
        # Skip connection for better gradient flow
        skip_output = self.skip_connection(last_hidden)
        
        # Combine outputs
        final_output = main_output + 0.1 * skip_output
        
        if self.output_activation:
            final_output = self.output_activation(final_output)
            
        return final_output

def train_gru_on_all(processed_dir: str, model_variant: str = "gru", 
                     batch_size: int = 64, epochs: int = 300, 
                     lr: float = 0.001, seq_len: int = 15):
    """
    Train GRU model with specific optimizations for position estimation
    """
    # Generate random seed based on current time
    random_seed = int(time.time() * 1000) % 100000
    print(f"Using random seed: {random_seed}")
    
    # Set random seeds
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Load data using dataset from config
    df = load_cir_data(processed_dir, filter_keyword=DATA_CONFIG['datasets'][0])
    print(f"Loaded {len(df)} data points from {DATA_CONFIG['datasets'][0]}")
    
    # Scale and create sequences
    X_seq, y_seq, x_scaler, y_scaler = scale_and_sequence(df, seq_len=seq_len)
    
    # Split data with random seed
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=random_seed, shuffle=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(random_seed + 1)  # Different seed for loader
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=batch_size,
        drop_last=False
    )
    
    # Create model with random initialization
    model = ImprovedGRUModel(input_dim=2, hidden_dim=128, num_layers=2, dropout=0.3)
    
    # Initialize weights with Xavier/Glorot initialization
    def init_weights(m):
        if isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    model.apply(init_weights)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer with adjusted parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-5,
        betas=(0.9, 0.999),  # Default Adam betas
        eps=1e-8  # Default Adam epsilon
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
        min_lr=1e-6  # Add minimum learning rate
    )
    
    train_loss_hist = []
    val_loss_hist = []
    best_val_loss = float('inf')
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
        y_val_actual, y_val_pred = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                val_batches += 1
                
                y_val_actual.extend(y_batch.cpu().numpy())
                y_val_pred.extend(preds.cpu().numpy())
        
        val_loss /= val_batches
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            # Check prediction diversity
            pred_std = np.std(y_val_pred)
            print(f"  Prediction std: {pred_std:.6f}")
    
    # Generate predictions on full dataset
    model.eval()
    with torch.no_grad():
        full_preds_scaled = model(X_seq.to(device)).cpu().numpy()
        full_targets_scaled = y_seq.numpy()
    
    # Inverse transform
    full_preds = y_scaler.inverse_transform(full_preds_scaled.reshape(-1, 1)).flatten()
    full_targets = y_scaler.inverse_transform(full_targets_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(np.mean((full_targets - full_preds) ** 2))
    
    print(f"\nFinal Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Prediction range: [{full_preds.min():.2f}, {full_preds.max():.2f}]")
    print(f"Target range: [{full_targets.min():.2f}, {full_targets.max():.2f}]")
    print(f"Prediction std: {np.std(full_preds):.4f}")
    print(f"Target std: {np.std(full_targets):.4f}")
    
    return {
        'r_actual': full_targets.tolist(),
        'r_pred': full_preds.tolist(),
        'train_loss': train_loss_hist,
        'val_loss': val_loss_hist,
        'rmse': rmse,
        'original_df_size': len(df),
        'sequence_size': len(full_targets),
        'seq_len': seq_len
    }

# Usage example
if __name__ == "__main__":
    # Example usage with your data
    results = train_gru_on_all(
        processed_dir="your_data_directory",
        batch_size=64,
        epochs=300,
        lr=0.001,
        seq_len=15
    )