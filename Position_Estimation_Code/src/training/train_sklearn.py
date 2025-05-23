from src.data.loader import load_cir_data, extract_features_and_target
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from src.models.model_registry import get_model
import numpy as np


def train_linear_on_all(processed_dir: str):
    df = load_cir_data(processed_dir, filter_keyword="FCPR-D1")

    X, y = extract_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = get_model("linear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Linear Regression RMSE: {rmse:.4f}")

    return y_test.tolist(), y_pred.tolist()


def train_svr_on_all(processed_dir: str):
    df = load_cir_data(processed_dir, filter_keyword="FCPR-D1")
    X, y = extract_features_and_target(df)

    # Scale features
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = get_model("svr", kernel="rbf", C=1.0, epsilon=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_scaled)

    rmse = root_mean_squared_error(y, y_pred)
    print(f"SVR RMSE: {rmse:.2f}")

    return y.tolist(), y_pred.tolist()