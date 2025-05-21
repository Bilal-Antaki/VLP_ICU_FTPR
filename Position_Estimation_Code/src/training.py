import time
from sklearn.model_selection import train_test_split
from src.data_processing import load_data, process_and_save_data
from src.models.linear_regression import LinearRegressor
from src.models.random_forest import RandomForestModel
from src.models.svr_model import SVRModel
from src.models.gradient_boosting import GradientBoostingModel
from src.models.lstm_model import LSTMRegressor
from src.evaluation import compute_rmse

def run_all(raw_path, processed_path, decimals=5):
    process_and_save_data(raw_path, processed_path, decimals)
    X, y = load_data(processed_path)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    models = {
        'LinearRegression': LinearRegressor(),
        'RandomForest': RandomForestModel(),
        'SVR': SVRModel(),
        'GradientBoosting': GradientBoostingModel(),
        'LSTM': LSTMRegressor(timesteps=1, features=1)
    }

    results = {}
    for name, model in models.items():
        start = time.time()
        if name == 'LSTM':
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        duration = time.time() - start
        preds = model.predict(X_test)
        rmse = compute_rmse(y_test, preds)
        results[name] = {'rmse': rmse, 'time': duration}

    print("Model Performance:")
    for name, res in results.items():
        print(f"{name}: RMSE = {res['rmse']:.5f}, Train Time = {res['time']:.2f}s")