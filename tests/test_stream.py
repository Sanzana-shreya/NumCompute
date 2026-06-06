import numpy as np
import pytest

from numcompute_stream.stream import stream_chunks, StreamingMetricTracker
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.tree import StreamingDecisionTreeClassifier


# For stream.py


class IdentityTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

    def partial_fit(self, X, y=None):
        return self


def test_stream_chunks_normal_case():
    # Normal chunking
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    chunks = list(stream_chunks(X, y, chunk_size=3))
    assert len(chunks) == 4


def test_stream_chunks_sizes():
    # Chunk sizes
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    chunks = list(stream_chunks(X, y, chunk_size=3))
    sizes = [chunk[0].shape[0] for chunk in chunks]

    assert sizes == [3, 3, 3, 1]


def test_stream_chunks_invalid_chunk_size():
    # Invalid chunk size
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    with pytest.raises(ValueError):
        list(stream_chunks(X, y, chunk_size=0))


def test_stream_chunks_invalid_X():
    # Invalid X
    X = np.arange(10)
    y = np.arange(10)

    with pytest.raises(ValueError):
        list(stream_chunks(X, y, chunk_size=2))


def test_stream_chunks_invalid_y():
    # Invalid y
    X = np.arange(20).reshape(10, 2)
    y = np.arange(20).reshape(10, 2)

    with pytest.raises(ValueError):
        list(stream_chunks(X, y, chunk_size=2))


def test_stream_chunks_shape_mismatch():
    # Shape mismatch
    X = np.arange(20).reshape(10, 2)
    y = np.arange(9)

    with pytest.raises(ValueError):
        list(stream_chunks(X, y, chunk_size=2))


def test_metric_tracker_case():
    # Metric tracking
    tracker = StreamingMetricTracker()
    tracker.update("accuracy", 0.8)
    tracker.update("accuracy", 0.9)

    assert tracker.get("accuracy") == [0.8, 0.9]


def test_metric_tracker_reset():
    # Reset history
    tracker = StreamingMetricTracker()
    tracker.update("accuracy", 0.8)
    tracker.reset()

    assert tracker.as_dict() == {}


def test_pipeline_partial_fit_case():
    # Pipeline partial fit
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    pipe = Pipeline([
        ("identity", IdentityTransformer()),
        ("tree", StreamingDecisionTreeClassifier(max_depth=3))
    ])

    pipe.partial_fit(X[:3], y[:3], classes=np.array([0, 1]))
    pipe.partial_fit(X[3:], y[3:])

    y_pred = pipe.predict(X)
    assert y_pred.shape == y.shape


def test_stream_chunks_shuffle_case():
    # Shuffle chunking
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    chunks = list(stream_chunks(X, y, chunk_size=5, shuffle=True, random_state=42))

    assert len(chunks) == 2
    assert chunks[0][0].shape[0] == 5
    assert chunks[1][0].shape[0] == 5