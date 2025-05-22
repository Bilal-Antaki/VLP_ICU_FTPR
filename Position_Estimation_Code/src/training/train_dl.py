import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.models.model_registry import get_model
from src.data.loader import load_cir_data
from src.data.preprocessing import scale_and_sequence

def train_lstm_on_all(processed_dir: str, seq_len: int = 10, batch_size: int = 32, epochs: int = 50, lr: float = 0.001):
    # Load and preprocess data
    df = load_cir_data(processed_dir)
    X_seq, y_seq, _ = scale_and_sequence(df, seq_len=seq_len)

    # Split into train, val, test (80/10/10)
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Wrap in DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # Initialize model
    model = get_model("lstm", input_dim=2, hidden_dim=64, num_layers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
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
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:03d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Evaluate on test set
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            test_loss += loss.item() * len(X_batch)
    test_loss /= len(test_loader.dataset)

    print(f"\nFinal Test Loss (RMSE): {test_loss**0.5:.4f}")
