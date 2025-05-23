import matplotlib.pyplot as plt

def plot_model_results(pl_values, r_actual, r_pred_linear, r_pred_lstm, train_loss, val_loss):
    """
    Plot two subplots:
    1. Actual vs Predicted r over PL
    2. Training and Validation Loss
    """

    # Align all to minimum length
    min_len = min(len(pl_values), len(r_actual), len(r_pred_linear), len(r_pred_lstm))
    pl_values = pl_values[:min_len]
    r_actual = r_actual[:min_len]
    r_pred_linear = r_pred_linear[:min_len]
    r_pred_lstm = r_pred_lstm[:min_len]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: r vs PL
    axs[0].plot(pl_values, r_actual, label="Actual", linestyle="--", marker="o")
    axs[0].plot(pl_values, r_pred_linear, label="Linear Predicted", linestyle="-", marker="x")
    axs[0].plot(pl_values, r_pred_lstm, label="LSTM Predicted", linestyle="-.", marker="s")
    axs[0].set_xlabel("PL")
    axs[0].set_ylabel("r (Distance)")
    axs[0].set_title("Predicted vs Actual r over PL")
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: Loss Curve
    axs[1].plot(train_loss, label="Train Loss")
    axs[1].plot(val_loss, label="Validation Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MSE Loss")
    axs[1].set_title("Loss Curve")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
