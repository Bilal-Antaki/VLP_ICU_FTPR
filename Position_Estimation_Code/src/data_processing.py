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
    """Process a single CSV file without headers to add polar radius and save it."""
    df = pd.read_csv(filepath, header=None)
    df.columns = ['X', 'Y', 'Measurement']  # Assign fixed column names

    # Compute polar radius
    df.insert(2, 'r', df.apply(lambda row: compute_radius(row['X'], row['Y'], decimals), axis=1))

    # Reorder to: X, Y, r, Measurement
    df = df[['X', 'Y', 'r', 'Measurement']]

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
