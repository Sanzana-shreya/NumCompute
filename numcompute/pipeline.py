class Pipeline:
    """
    Chain preprocessing transformers and a final estimator.

    Transformers must implement:
    fit(X)
    transform(X)

    Final estimator must implement:
    fit(X, y)
    predict(X)
    """

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline requires at least one step")

        self.steps = steps

    def fit(self, X, y=None):
        X_current = X

        # Fit and transform all steps except the last one
        for name, step in self.steps[:-1]:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise ValueError(f"Step '{name}' must implement fit and transform")

            step.fit(X_current)
            X_current = step.transform(X_current)

        # Fit final step
        last_name, last_step = self.steps[-1]

        if not hasattr(last_step, "fit"):
            raise ValueError(f"Final step '{last_name}' must implement fit")

        if y is not None:
            last_step.fit(X_current, y)
        else:
            last_step.fit(X_current)

        return self

    def transform(self, X):
        X_current = X

        # Transform only transformer steps
        for name, step in self.steps:
            if hasattr(step, "transform"):
                X_current = step.transform(X_current)
            else:
                break

        return X_current

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        X_current = X

        # Apply all transformers except final estimator
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(f"Step '{name}' must implement transform")

            X_current = step.transform(X_current)

        last_name, last_step = self.steps[-1]

        if not hasattr(last_step, "predict"):
            raise ValueError(f"Final step '{last_name}' must implement predict")

        return last_step.predict(X_current)