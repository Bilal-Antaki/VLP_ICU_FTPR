from src.training.train_sklearn import train_linear_on_all
from src.training.train_dl import train_lstm_on_all
import pandas as pd
from src.data.loader import load_cir_data
from src.utils.visualizations import plot_model_results
from src.training.train_sklearn import train_svr_on_all

PROCESSED_DATA_DIR = r"C:\Dev\Python\Position_Estimation_Code\data\processed"

if __name__ == "__main__":

    # Load original r values from full processed dataframe
    df = load_cir_data(PROCESSED_DATA_DIR, filter_keyword="FCPR-D1")
    r_full = df["r"].tolist()
    pl_full = df["r"].tolist()

    # Slice from the 5th entry onward to match LSTM/SVR prediction window
    r_actual = r_full[4:]
    pl_values = pl_full[4:]

    # Train Linear
    r_actual_linear, r_pred_linear = train_linear_on_all(PROCESSED_DATA_DIR)

    # Train SVR
    r_actual_svr, r_pred_svr = train_svr_on_all(PROCESSED_DATA_DIR)


    # Train LSTM (simple batch mode)
    r_actual_lstm, r_pred_lstm, train_loss, val_loss = train_lstm_on_all(PROCESSED_DATA_DIR)

    # Align lengths
    min_len = min(len(r_actual), len(r_pred_linear), len(r_pred_lstm))
    r_actual = r_actual[-min_len:]
    r_pred_linear = r_pred_linear[-min_len:]
    r_pred_lstm = r_pred_lstm[-min_len:]

    print(f"len(r_actual): {len(r_actual)}")
    print(f"len(r_pred_linear): {len(r_pred_linear)}")
    print(f"len(r_pred_lstm): {len(r_pred_lstm)}")

    # Show results
    df_compare = pd.DataFrame({
        "r_actual": r_actual,
        "r_pred_linear": r_pred_linear,
        "r_pred_lstm": r_pred_lstm,
        "r_pred_svr": r_pred_svr[:min_len]
    }).round(2)

    #print("\n[r_actual, r_pred_linear, r_pred_lstm]:")
    print(df_compare.head(min_len))

    # plot_model_results(pl_values, r_actual, r_pred_linear, r_pred_lstm, train_loss, val_loss)
