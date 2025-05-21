from src.data_processing import process_all_files

if __name__ == "__main__":
    decimals = int(input("Enter number of decimal places for polar radius r: "))
    process_all_files(
        raw_dir=r"C:\Dev\Python\Position_Estimation_Code\data\raw",
        processed_dir=r"C:\Dev\Python\Position_Estimation_Code\data\processed",
        decimals=decimals
    )
