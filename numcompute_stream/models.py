import numpy as np

class ZeroRClassifier:
    """
    Predicts the most frequent class.
    """

    def __init__(self):
        self.most_common = None

    def fit(self, X, y):
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be 1D")

        values, counts = np.unique(y, return_counts=True)
        self.most_common = values[np.argmax(counts)]

        return self

    def predict(self, X):
        if self.most_common is None:
            raise ValueError("Model has not been fitted yet")

        X = np.asarray(X)

        return np.full(X.shape[0], self.most_common)
    

class ZeroRRegressor:
    """
    Predicts the mean value.
    """

    def __init__(self):
        self.mean_value = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError("y must be 1D")

        self.mean_value = np.mean(y)

        return self

    def predict(self, X):
        if self.mean_value is None:
            raise ValueError("Model has not been fitted yet")

        X = np.asarray(X)

        return np.full(X.shape[0], self.mean_value)
