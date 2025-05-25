import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Add dropout for regularization
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Add a more complex output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Take the last timestep output
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        output = self.fc(last_output)
        return output.squeeze(-1)  # [batch]

def build_lstm_model(**kwargs):
    return LSTMRegressor(**kwargs)
