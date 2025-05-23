import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=85, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [batch, 1, input_dim]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out).squeeze()

def build_lstm_model(**kwargs):
    return LSTMRegressor(**kwargs)
