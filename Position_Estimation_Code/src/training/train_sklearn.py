import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.loader import load_cir_data


X_train, X_test, y_train, y_test = load_cir_data(
    data_dir="C:/Dev/Python/Position_Estimation_Code/data/processed",
    features=["CIR"], 
    target="r"
)
