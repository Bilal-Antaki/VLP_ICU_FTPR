from src.models.model_registry import get_model
from src.data.loader import load_cir_data, extract_features_and_target
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def train_linear_on_all(processed_dir: str):
    df = load_cir_data(processed_dir)
    X, y = extract_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = get_model("linear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Linear Regression RMSE: {rmse:.4f}")

    return y_test.tolist(), y_pred.tolist()
