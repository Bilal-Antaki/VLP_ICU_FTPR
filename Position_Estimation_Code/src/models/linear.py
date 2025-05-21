from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LinearRegressor:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        metrics = {
            'MSE': mean_squared_error(y_test, predictions),
            'MAE': mean_absolute_error(y_test, predictions),
            'R2': r2_score(y_test, predictions)
        }
        return metrics