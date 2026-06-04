import numpy as np


def stream_chunks(
    X: np.ndarray,
    y: np.ndarray,
    chunk_size: int = 32,
    shuffle: bool = False,
    random_state: int | None = None
):
    """
    Yield data in chunks to simulate streaming.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    chunk_size : int
        Number of rows per chunk.
    shuffle : bool
        Whether to shuffle before splitting into chunks.
    random_state : int, optional
        Seed for reproducibility.

    Yields
    ------
    X_chunk : np.ndarray
    y_chunk : np.ndarray
    """
    if X is None or y is None:
        raise ValueError("X and y cannot be None")

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be 2D")

    if y.ndim != 1:
        raise ValueError("y must be 1D")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    indices = np.arange(X.shape[0])

    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    for start in range(0, X.shape[0], chunk_size):
        end = start + chunk_size
        yield X[start:end], y[start:end]


class StreamingMetricTracker:
    """
    Track streaming metrics over time.
    """

    def __init__(self):
        self.history = {}

    def update(self, metric_name: str, value: float):
        """
        Add metric value to history.
        """
        if metric_name not in self.history:
            self.history[metric_name] = []

        self.history[metric_name].append(float(value))

    def get(self, metric_name: str):
        """
        Get metric history.
        """
        return self.history.get(metric_name, [])

    def reset(self):
        """
        Clear all stored history.
        """
        self.history = {}

    def as_dict(self):
        """
        Return metric history dictionary.
        """
        return self.history