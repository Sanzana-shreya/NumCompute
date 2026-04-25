#simple pipeline to apply multiple preprocessing steps in sequence
class Pipeline:
    #stores the list of steps (like scaler, encoder, etc.)
    def __init__(self, steps):
        self.steps = steps

    #fits each step and passes transformed data to the next step
    def fit(self, X):
        for name, step in self.steps:
            step.fit(X)
            X = step.transform(X)
        return X

    #applies all steps sequentially without fitting again 
    def transform(self, X):
        for name, step in self.steps:
            X = step.transform(X)
        return X
    
    #fits and transforms the data in one call
    def fit_transform(self, X):
        for name, step in self.steps:
            step.fit(X)
            X = step.transform(X)
        return X
    
    #prediction just means applying the saved transformations
    def predict(self, X):
        return self.transform(X)
    