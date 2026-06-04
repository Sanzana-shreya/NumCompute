class Pipeline:
    """
    Chain preprocessing transformers and a final estimator.

    Transformers must implement:
    fit(X)
    transform(X)

    Optional streaming support:
    partial_fit(X) or partial_fit(X, y)

    Final estimator must implement:
    fit(X, y)
    predict(X)

    Optional streaming support:
    partial_fit(X, y)
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

    def partial_fit(self, X, y=None, classes=None):
        """
        Incrementally fit the pipeline on a chunk of data.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like, optional
            Target values.
        classes : array-like, optional
            Class labels for incremental classifiers on the first call.

        Returns
        -------
        self
            Fitted pipeline.
        """
        X_current = X

        # Partial fit or transform intermediate steps
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(f"Step '{name}' must implement transform")

            if hasattr(step, "partial_fit"):
                # Try partial_fit(X, y), else partial_fit(X)
                try:
                    if y is not None:
                        step.partial_fit(X_current, y)
                    else:
                        step.partial_fit(X_current)
                except TypeError:
                    step.partial_fit(X_current)

            elif hasattr(step, "fit"):
                # fallback for non-streaming transformers:
                # fit once, then reuse transform
                if not hasattr(step, "_pipeline_fitted"):
                    step.fit(X_current)
                    step._pipeline_fitted = True
            else:
                raise ValueError(f"Step '{name}' must implement fit or partial_fit")

            X_current = step.transform(X_current)

        # Partial fit final estimator
        last_name, last_step = self.steps[-1]

        if hasattr(last_step, "partial_fit"):
            try:
                if classes is not None:
                    last_step.partial_fit(X_current, y, classes=classes)
                else:
                    last_step.partial_fit(X_current, y)
            except TypeError:
                if y is not None:
                    last_step.partial_fit(X_current, y)
                else:
                    last_step.partial_fit(X_current)

        elif hasattr(last_step, "fit"):
            if y is not None:
                last_step.fit(X_current, y)
            else:
                last_step.fit(X_current)
        else:
            raise ValueError(f"Final step '{last_name}' must implement fit or partial_fit")

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