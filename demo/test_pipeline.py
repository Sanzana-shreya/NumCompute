import sys
import os
import pytest

# Add parent directory to path to import numcompute package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numcompute.pipeline import Pipeline


# Dummy step to simulate a transformer
class DummyStep:
    def __init__(self):
        self.factor = 2

    def fit(self, X):
        # Simulates fitting (no actual learning)
        return X

    def transform(self, X):
        # Multiplies each element by a constant factor
        return [x * self.factor for x in X]


def test_pipeline_basic():
    # Define pipeline steps
    steps = [
        ("step1", DummyStep()),
        ("step2", DummyStep())
    ]

    pipeline = Pipeline(steps)

    # Input data
    X = [1, 2, 3]

    # Fit pipeline
    pipeline.fit(X)

    # Predict (apply transformations)
    result = pipeline.predict(X)

    # Each step multiplies by 2 → total effect = x * 2 * 2 = x * 4
    assert result == [4, 8, 12]


def test_pipeline_without_fit():
    # Define pipeline with single step
    steps = [
        ("step1", DummyStep())
    ]

    pipeline = Pipeline(steps)

    # Input data
    X = [1, 2, 3]

    # Expect error if predict is called before fit
    with pytest.raises(Exception):
        pipeline.predict(X)