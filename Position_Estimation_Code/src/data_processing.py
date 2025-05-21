import os
import glob
import pandas as pd
import numpy as np

def process_and_save_data(raw_path: str, processed_path: str, decimals: int = 5):
    os.makedirs(processed_path, exist_ok=True)
    file_paths = glob.glob(f"{raw_path}/*.csv")
    for fp in file_paths:
        df = pd.read_csv(fp, header=None)
        # Compute new R (column 3)
        R = np.hypot(df.iloc[:, 0], df.iloc[:, 1]).round(decimals)
        # Shift old Output (col3) to col4
        df.insert(3, 'OLD', df.iloc[:, 2])
        # Replace col3 with R
        df.iloc[:, 2] = R
        # Save processed file
        filename = os.path.basename(fp)
        df.to_csv(os.path.join(processed_path, filename), header=False, index=False)

def load_data(processed_path: str):
    file_paths = glob.glob(f"{processed_path}/*.csv")
    all_X, all_y = [], []
    for fp in file_paths:
        df = pd.read_csv(fp, header=None)
        R = df.iloc[:, 2]
        CIR = df.iloc[:, 3]
        all_X.append(CIR.values.reshape(-1, 1))
        all_y.append(R.values)
    X = np.vstack(all_X)
    y = np.hstack(all_y)
    return X, y