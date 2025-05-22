import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze()
    
def build_lstm_model(**kwargs):
    return LSTMRegressor(**kwargs)
