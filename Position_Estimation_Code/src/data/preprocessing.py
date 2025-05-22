import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def sequence_split(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    return np.array(X_seq), np.array(y_seq)

def scale_and_sequence(df, seq_len=10, features=['PL', 'RMS'], target='r'):
    """
    Scale features and convert into sequences suitable for LSTM input.
    """
    X = df[features].values
    y = df[target].values

    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Sequence
    X_seq, y_seq = sequence_split(X_scaled, y, seq_len)
    return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32), scaler
