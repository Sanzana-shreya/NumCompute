import sys
import os
import pytest

# Add parent directory to path to import numcompute package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numcompute.pipeline import Pipeline


# Dummy step to simulate a transformer
class DummyTransformer:
    def __init__(self):
        self.factor = 2

    def fit(self, X):
        # Simulates fitting (no actual learning)
        return self

    def transform(self, X):
        # Multiplies each element by a constant factor
        return [x * self.factor for x in X]


# Dummy estimator to simulate final prediction step
class DummyEstimator:
    def fit(self, X, y=None):
        # Simulates fitting (no actual learning)
        return self

    def predict(self, X):
        # Returns transformed input directly
        return X


def test_pipeline_basic():
    # Define pipeline steps
    steps = [
        ("step1", DummyTransformer()),
        ("step2", DummyEstimator())
    ]

    pipeline = Pipeline(steps)

    # Input data
    X = [1, 2, 3]

    # Fit pipeline
    pipeline.fit(X)

    # Predict
    result = pipeline.predict(X)

    # Transformer multiplies by 2 → final output = x * 2
    assert result == [2, 4, 6]


def test_pipeline_without_fit():
    # Define pipeline with transformer + estimator
    steps = [
        ("step1", DummyTransformer()),
        ("step2", DummyEstimator())
    ]

    pipeline = Pipeline(steps)

    # Input data
    X = [1, 2, 3]

    # In this simple dummy setup, predict still works because transformer has no fitted state
    result = pipeline.predict(X)

    assert result == [2, 4, 6]


def test_pipeline_empty_steps():
    # Pipeline should fail if no steps are provided
    with pytest.raises(ValueError):
        Pipeline([])


def test_pipeline_invalid_transformer():
    # Invalid transformer without transform
    class BadTransformer:
        def fit(self, X):
            return self

    steps = [
        ("bad", BadTransformer()),
        ("final", DummyEstimator())
    ]

    pipeline = Pipeline(steps)

    with pytest.raises(ValueError):
        pipeline.fit([1, 2, 3])


def test_pipeline_invalid_estimator():
    # Invalid final estimator without predict
    class BadEstimator:
        def fit(self, X, y=None):
            return self

    steps = [
        ("step1", DummyTransformer()),
        ("bad", BadEstimator())
    ]

    pipeline = Pipeline(steps)
    pipeline.fit([1, 2, 3])

    with pytest.raises(ValueError):
        pipeline.predict([1, 2, 3])