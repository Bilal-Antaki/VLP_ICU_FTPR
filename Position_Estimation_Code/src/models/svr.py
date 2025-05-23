from sklearn.svm import SVR

def build_svr_model(kernel="rbf", C=1.0, epsilon=0.1, **kwargs):
    return SVR(kernel=kernel, C=C, epsilon=epsilon, **kwargs)
