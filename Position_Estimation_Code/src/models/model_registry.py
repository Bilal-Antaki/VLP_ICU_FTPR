from .linear import build_linear_model
#from .svr import build_svr_model
#from .knn import build_knn_model
#from .mlp import build_mlp_model

# Optional: import DL models if needed
# from .lstm import build_lstm_model
# from .rnn import build_rnn_model

MODEL_REGISTRY = {
    "linear": build_linear_model,
    #"svr": build_svr_model,
    #"knn": build_knn_model,
    #"mlp": build_mlp_model,
    # "rnn": build_rnn_model,
    # "lstm": build_lstm_model,
}

def get_model(name: str, **kwargs):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    return MODEL_REGISTRY[name](**kwargs)
