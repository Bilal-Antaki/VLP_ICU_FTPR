from src.training.train_sklearn import train_linear_on_all
from src.training.train_dl import train_lstm_on_all
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DATA_DIR = r"C:\Dev\Python\Position_Estimation_Code\data\processed"

if __name__ == "__main__":
    # Train Linear
    r_actual, r_pred_linear = train_linear_on_all(PROCESSED_DATA_DIR)

    # Train LSTM (simple batch mode)
    r_actual_lstm, r_pred_lstm, train_loss, val_loss = train_lstm_on_all(PROCESSED_DATA_DIR)

    # Align lengths
    min_len = min(len(r_actual), len(r_actual_lstm))
    r_actual = r_actual[-min_len:]
    r_pred_linear = r_pred_linear[-min_len:]
    r_pred_lstm = r_pred_lstm[-min_len:]

    # Show results
    df_compare = pd.DataFrame({
        "r_actual": r_actual,
        "r_pred_linear": r_pred_linear,
        "r_pred_lstm": r_pred_lstm
    }).round(2)

    #print("\n[r_actual, r_pred_linear, r_pred_lstm]:")
    print(df_compare.head(min_len))

    # Plot loss
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("LSTM Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()
