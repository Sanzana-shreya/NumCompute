import numpy as np

from numcompute.benchmarking import benchmark_streaming_model
from numcompute.tree import StreamingDecisionTreeClassifier


def test_benchmark_streaming_model():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    classes = np.array([0, 1])

    model = StreamingDecisionTreeClassifier(max_depth=3)

    result = benchmark_streaming_model(model, X, y, classes, chunk_size=2)

    assert isinstance(result, dict)
    assert "time" in result
    assert "memory_mb" in result
    assert "accuracy_history" in result