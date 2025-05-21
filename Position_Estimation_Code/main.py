from src.training import run_all

if __name__ == '__main__':
    # Example: process data and train models with 5-decimal precision
    folder_path = r"C:\Dev\Python\Position_Estimation_Code\CIRs\test"
    decimals = 5
    run_all(folder_path, decimals)