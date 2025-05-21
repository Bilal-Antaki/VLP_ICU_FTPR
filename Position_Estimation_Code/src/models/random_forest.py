from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)