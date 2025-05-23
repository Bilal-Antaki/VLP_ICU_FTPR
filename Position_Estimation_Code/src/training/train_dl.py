import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.models.model_registry import get_model
from src.data.loader import load_cir_data
from src.data.preprocessing import scale_and_sequence
import numpy as np

def train_lstm_on_all(processed_dir: str, batch_size: int = 10, epochs: int = 55, lr: float = 0.001):
    seq_len=4
    df = load_cir_data(processed_dir, filter_keyword="FCPR-D1")

    X_seq, y_seq, x_scaler, y_scaler = scale_and_sequence(df, seq_len=seq_len)

    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = get_model("lstm", input_dim=2, hidden_dim=55, num_layers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_hist, val_loss_hist = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        y_val_actual, y_val_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(X_batch)
                y_val_actual.extend(y_batch.cpu().numpy())
                y_val_pred.extend(preds.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # --- Predict on full input sequence set (not just val) ---
    model.eval()
    with torch.no_grad():
        full_preds = model(X_seq.to(device)).cpu().numpy()
        full_targets = y_seq.numpy()

    # Inverse transform both
    full_preds = y_scaler.inverse_transform(full_preds.reshape(-1, 1)).flatten()
    full_targets = y_scaler.inverse_transform(full_targets.reshape(-1, 1)).flatten()

    rmse = np.sqrt(val_loss)
    print(f"\nFinal Validation RMSE: {rmse:.4f}")
    print("LSTM predictions:", len(full_preds))
    print("Expected:", len(df) - seq_len)

    return full_targets.tolist(), full_preds.tolist(), train_loss_hist, val_loss_hist
