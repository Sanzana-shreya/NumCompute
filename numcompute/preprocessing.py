import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

class OneHotEncoder:
    def __init__(self):
        self.categories = None

    def fit(self, X):
        self.categories = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self
    
    def transform(self, X):

        if self.categories is None:
            raise ValueError("The encoder has not been fitted yet.")
        encoded_columns = []

        for i in range(X.shape[1]):
            col = X[:, i]
            categories = self.categories[i]

            one_hot = (col[:, None] == categories).astype(int)
            encoded_columns.append(one_hot)

        return np.hstack(encoded_columns)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)