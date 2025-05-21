import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_cir_data(data_dir, features, target='r', test_size=0.2, return_type='numpy'):
    """
    Load and concatenate processed CIR CSV files, extract features and target.

    Args:
        data_dir (str): Path to processed CSV directory
        features (list of str): Feature column names (e.g., ['PL'] or ['PL', 'RMS'])
        target (str): Target column name (default: 'r')
        test_size (float): Train/test split ratio
        return_type (str): 'numpy' or 'pandas'

    Returns:
        X_train, X_test, y_train, y_test (as numpy arrays or pandas DataFrames)
    """
    all_data = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)

            # Ensure columns exist
            required_cols = set(features + [target])
            if not required_cols.issubset(df.columns):
                continue

            all_data.append(df[features + [target]])

    if not all_data:
        raise ValueError("No valid data files found in directory.")

    data = pd.concat(all_data, ignore_index=True)
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if return_type == 'numpy':
        return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
    else:
        return X_train, X_test, y_train, y_test
