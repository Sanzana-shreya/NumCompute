import numpy as np
from typing import Tuple


def _validate_X_y(
    X: np.ndarray,
    y: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Validate feature matrix X and optional target y.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    y : np.ndarray of shape (n_samples,), optional
        Target labels.

    Returns
    -------
    X : np.ndarray
        Validated feature matrix as float.
    y : np.ndarray or None
        Validated target array.

    Raises
    ------
    ValueError
        If shapes are invalid or sizes mismatch.
    """
    if X is None:
        raise ValueError("X cannot be None")

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if y is not None:
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

    return X, y


def _gini(y: np.ndarray) -> float:
    """
    Compute Gini impurity.

    Parameters
    ----------
    y : np.ndarray
        Class labels.

    Returns
    -------
    float
        Gini impurity.
    """
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs ** 2))


def _entropy(y: np.ndarray) -> float:
    """
    Compute entropy.

    Parameters
    ----------
    y : np.ndarray
        Class labels.

    Returns
    -------
    float
        Entropy value.
    """
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _majority_class(y: np.ndarray):
    """
    Return majority class label.

    Parameters
    ----------
    y : np.ndarray
        Class labels.

    Returns
    -------
    label
        Most frequent class.
    """
    if len(y) == 0:
        raise ValueError("Cannot compute majority class for empty array")

    labels, counts = np.unique(y, return_counts=True)
    return labels[np.argmax(counts)]


class _Node:
    """
    Internal tree node.
    """

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.is_leaf = True


class StreamingDecisionTreeClassifier:
    """
    Streaming-compatible decision tree classifier.

    Notes
    -----
    This implementation supports chunk-wise updates using partial_fit().
    Internally it stores seen data and rebuilds the tree after each update.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini"
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion

        self.root = None
        self.X_seen = None
        self.y_seen = None
        self.classes_ = None
        self.n_features_in_ = None

    def _impurity(self, y: np.ndarray) -> float:
        """
        Select impurity function.
        """
        if self.criterion == "gini":
            return _gini(y)
        elif self.criterion == "entropy":
            return _entropy(y)
        else:
            raise ValueError("criterion must be 'gini' or 'entropy'")

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Find the best split for the current node.

        Returns
        -------
        best_feature : int or None
        best_threshold : float or None
        """
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        parent_impurity = self._impurity(y)

        for feature in range(n_features):
            column = X[:, feature]

            # Ignore NaNs for threshold search
            valid = ~np.isnan(column)
            column_valid = column[valid]
            y_valid = y[valid]

            if len(column_valid) < self.min_samples_split:
                continue

            unique_vals = np.unique(column_valid)

            if len(unique_vals) <= 1:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask = valid & (column <= threshold)
                right_mask = valid & (column > threshold)

                y_left = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                n_left = len(y_left)
                n_right = len(y_right)
                n_total = n_left + n_right

                child_impurity = (
                    (n_left / n_total) * self._impurity(y_left)
                    + (n_right / n_total) * self._impurity(y_right)
                )

                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """
        Recursively build the tree.
        """
        node = _Node()
        node.value = _majority_class(y)

        if len(np.unique(y)) == 1:
            return node

        if depth >= self.max_depth:
            return node

        if len(y) < self.min_samples_split:
            return node

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return node

        column = X[:, feature]
        valid = ~np.isnan(column)

        left_mask = valid & (column <= threshold)
        right_mask = valid & (column > threshold)

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return node

        node.is_leaf = False
        node.feature_index = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit decision tree on full dataset.
        """
        X, y = _validate_X_y(X, y)

        self.X_seen = X.copy()
        self.y_seen = y.copy()
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.root = self._build_tree(self.X_seen, self.y_seen)

        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray | None = None
    ):
        """
        Incrementally update the model with a new chunk.
        """
        X, y = _validate_X_y(X, y)

        if self.X_seen is None:
            self.X_seen = X.copy()
            self.y_seen = y.copy()
            self.n_features_in_ = X.shape[1]
        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError("Feature count does not match previous chunks")

            self.X_seen = np.vstack([self.X_seen, X])
            self.y_seen = np.concatenate([self.y_seen, y])

        if classes is not None:
            self.classes_ = np.asarray(classes)
        else:
            self.classes_ = np.unique(self.y_seen)

        self.root = self._build_tree(self.X_seen, self.y_seen)

        return self

    def _predict_one(self, x: np.ndarray, node: _Node):
        """
        Predict one sample.
        """
        if node.is_leaf:
            return node.value

        val = x[node.feature_index]

        if np.isnan(val):
            return node.value

        if val <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        """
        X, _ = _validate_X_y(X)

        if self.root is None:
            raise ValueError("Model has not been fitted yet")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match fitted model")

        y_pred = np.array([self._predict_one(x, self.root) for x in X])
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Notes
        -----
        This simplified version returns one-hot probabilities
        based on predicted class.
        """
        X, _ = _validate_X_y(X)

        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet")

        preds = self.predict(X)
        probs = np.zeros((len(preds), len(self.classes_)))

        for i, pred in enumerate(preds):
            idx = np.where(self.classes_ == pred)[0][0]
            probs[i, idx] = 1.0

        return probs