import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the ICU-D1_CIR.csv file
processed_dir = r"C:\Dev\Python\Position_Estimation_Code\data\processed"
filename = "ICU-D1_CIR.csv"

# Some systems may save ICU with RTL characters (e.g. "\u200fICU")
files = os.listdir(processed_dir)
icu_d1_file = next((f for f in files if "ICU" in f and "D1" in f), None)

# Load and plot if file is found
if icu_d1_file:
    filepath = os.path.join(processed_dir, icu_d1_file)
    df = pd.read_csv(filepath)

    # Plot PL vs r
    plt.scatter(df['r'], df['PL'], alpha=0.7)
    plt.xlabel("Distance r (m)")
    plt.ylabel("Path Loss (PL)")
    plt.title(f"PL vs distance for D1 in ICU")
    plt.grid(True)
    plt.show()
else:
    print("ICU-D1 file not found.")
