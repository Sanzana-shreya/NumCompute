import numpy as np
from .tree import StreamingDecisionTreeClassifier, _validate_X_y


class StreamingRandomForestClassifier:
    """
    Streaming-compatible random forest classifier.

    Notes
    -----
    Uses multiple streaming decision trees and majority voting.
    Each chunk can be bootstrapped before updating each tree.
    """

    def __init__(
        self,
        n_estimators: int = 5,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini",
        bootstrap: bool = True,
        random_state: int | None = None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_ = []
        self.classes_ = None
        self.n_features_in_ = None
        self._rng = np.random.RandomState(random_state)

        for _ in range(self.n_estimators):
            self.estimators_.append(
                StreamingDecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    criterion=self.criterion
                )
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit all trees on full data.
        """
        return self.partial_fit(X, y, classes=np.unique(y))

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray | None = None
    ):
        """
        Incrementally fit the forest on a new chunk.
        """
        X, y = _validate_X_y(X, y)

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match previous chunks")

        if classes is not None:
            self.classes_ = np.asarray(classes)
        elif self.classes_ is None:
            self.classes_ = np.unique(y)

        n_samples = X.shape[0]

        for tree in self.estimators_:
            if self.bootstrap:
                idx = self._rng.randint(0, n_samples, size=n_samples)
                X_chunk = X[idx]
                y_chunk = y[idx]
            else:
                X_chunk = X
                y_chunk = y

            tree.partial_fit(X_chunk, y_chunk, classes=self.classes_)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using majority voting.
        """
        X, _ = _validate_X_y(X)

        if self.n_features_in_ is None:
            raise ValueError("Model has not been fitted yet")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match fitted model")

        all_preds = np.array([tree.predict(X) for tree in self.estimators_]).T

        final_preds = []
        for row in all_preds:
            labels, counts = np.unique(row, return_counts=True)
            final_preds.append(labels[np.argmax(counts)])

        return np.array(final_preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities as average of tree probabilities.
        """
        X, _ = _validate_X_y(X)

        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet")

        probs = np.mean(
            [tree.predict_proba(X) for tree in self.estimators_],
            axis=0
        )

        return probs