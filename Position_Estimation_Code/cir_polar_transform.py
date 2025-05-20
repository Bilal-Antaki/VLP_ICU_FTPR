import glob
import pandas as pd
import numpy as np

def process_cir_csv_files(folder_path: str, decimals: int):
    """
    1. Reads the CSV, grabs columns 0 (X), 1 (Y), and last (Output).
    2. Computes R = sqrt(X^2 + Y^2), rounded to `decimals`.
    3. Rewrites the CSV in-place as [X, Y, R, Output].
    
    Parameters:
    -----------
    folder_path : str
    decimals : int
        Number of decimal places for the computed R column.
    """
    file_paths = glob.glob(f"{folder_path}/*.csv")

    for file_path in file_paths:
        
        df = pd.read_csv(file_path, header=None)
        

        X = df.iloc[:, 0]
        Y = df.iloc[:, 1]
        Output = df.iloc[:, -1]
        
        R = np.hypot(X, Y).round(decimals)
        
        
        clean_df = pd.DataFrame({
            'X': X,
            'Y': Y,
            'R': R,
            'Output': Output
        })
        
        clean_df.to_csv(file_path, index=False, header=False)

if __name__ == "__main__":
    folder_path = r"C:\Dev\Python\Position_Estimation_Code\CIRs"
    decimals = 2  # change to whatever precision you need
    
    process_cir_csv_files(folder_path, decimals)
