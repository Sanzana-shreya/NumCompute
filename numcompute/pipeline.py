class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X):
        for name, step in self.steps:
            step.fit(X)
            X = step.transform(X)
        return X
    
    def transform(self, X):
        for name, step in self.steps:
            X = step.transform(X)
        return X
    
    def fit_transform(self, X):
        for name, step in self.steps:
            step.fit(X)
            X = step.transform(X)
        return X
    
    def predict(self, X):
        return self.transform(X)
    