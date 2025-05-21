from src.training.train_sklearn import train_linear_on_all
from src.data.data_processing import process_all_pairs

RAW_DATA_DIR = r"C:\Dev\Python\Position_Estimation_Code\data\raw"
PROCESSED_DATA_DIR = r"C:\Dev\Python\Position_Estimation_Code\data\processed"

if __name__ == "__main__":
    # You can change the decimal precision here
    decimals_for_radius = 2
    process_all_pairs(RAW_DATA_DIR, PROCESSED_DATA_DIR, decimals=decimals_for_radius)
    train_linear_on_all(PROCESSED_DATA_DIR)