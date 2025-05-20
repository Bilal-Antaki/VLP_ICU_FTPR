import glob
import pandas as pd
import numpy as np

def process_cir_files(folder_path):
    """
    Processes and fully overwrites all .xlsx files:
    - Reads columns X, Y, Output
    - Computes R = sqrt(X^2 + Y^2)
    - Inserts R as 3rd column, shifts Output to 4th
    - Rewrites the entire file
    """
    file_paths = glob.glob(f"{folder_path}/*.xlsx")

    for file_path in file_paths:
        # Read just the data: no headers, first three columns only
        df = pd.read_excel(file_path, header=None, usecols=[0, 1, 2])
        df.columns = ['X', 'Y', 'Output']

        # Compute R and insert into DataFrame
        df['R'] = np.sqrt(df['X']**2 + df['Y']**2)
        df = df[['X', 'Y', 'R', 'Output']]

        # Fully overwrite the Excel file (removes all old sheets)
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, index=False, header=False)
        
        print(f"Updated: {file_path}")

if __name__ == "__main__":
    folder_path = "C:\Dev\Python\Position_Estimation_Code\CIRs"
    process_cir_files(folder_path)
