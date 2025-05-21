import os
import pandas as pd
import numpy as np

# Constants for directories
RAW_DATA_DIR = r"C:\Dev\Python\Position_Estimation_Code\data\raw"
PROCESSED_DATA_DIR = r"C:\Dev\Python\Position_Estimation_Code\data\processed"

def compute_radius(x: float, y: float, decimals: int) -> float:
    """Compute polar radius from x and y with rounding."""
    return round(np.sqrt(x**2 + y**2), decimals)

def process_file(filepath: str, output_dir: str, decimals: int) -> None:
    """Process a single CSV file to add polar radius and save it."""
    df = pd.read_csv(filepath)

    if not {'X', 'Y'}.issubset(df.columns):
        raise ValueError(f"File {filepath} must contain 'X' and 'Y' columns")

    # Determine measurement column (PL or RMS)
    measurement_cols = [col for col in df.columns if col not in ['X', 'Y']]
    if len(measurement_cols) != 1:
        raise ValueError(f"File {filepath} must have exactly one measurement column (PL or RMS)")
    
    measurement_col = measurement_cols[0]

    # Compute polar radius
    df.insert(2, 'r', df.apply(lambda row: compute_radius(row['X'], row['Y'], decimals), axis=1))

    # Reorder columns to: X, Y, r, measurement
    df = df[['X', 'Y', 'r', measurement_col]]

    # Save processed file
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)

def process_all_files(raw_dir: str, processed_dir: str, decimals: int) -> None:
    """Process all CSV files in the raw data directory."""
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for file in os.listdir(raw_dir):
        if file.endswith(".csv"):
            full_path = os.path.join(raw_dir, file)
            try:
                process_file(full_path, processed_dir, decimals)
                print(f"Processed: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
