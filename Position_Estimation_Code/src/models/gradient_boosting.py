from sklearn.ensemble import GradientBoostingRegressor
from .base_model import BaseModel

class GradientBoostingModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)