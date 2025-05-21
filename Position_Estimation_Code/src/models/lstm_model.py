import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from .base_model import BaseModel

class LSTMRegressor(BaseModel):
    def __init__(self, timesteps=1, features=1, units=50):
        self.timesteps = timesteps
        self.features = features
        self.units = units
        self.model = Sequential([
            LSTM(units, input_shape=(timesteps, features)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X_train, y_train, epochs=20, batch_size=16):
        Xr = X_train.reshape((X_train.shape[0], self.timesteps, self.features))
        self.model.fit(Xr, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        Xr = X.reshape((X.shape[0], self.timesteps, self.features))
        return self.model.predict(Xr).flatten()