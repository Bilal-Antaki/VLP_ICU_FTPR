import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def sequence_split(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)


def scale_and_sequence(df, seq_len=4, features=['PL', 'RMS'], target='r'):
    X = df[features].values
    y = df[[target]].values  # keep 2D for scaler

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y).flatten()  # back to 1D

    X_seq, y_seq = sequence_split(X_scaled, y_scaled, seq_len)

    return (
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(y_seq, dtype=torch.float32),
        x_scaler,
        y_scaler
    )

