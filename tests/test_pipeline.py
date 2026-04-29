import numpy as np
import pytest

from numcompute.pipeline import Pipeline


class DummyTransformer:
    def __init__(self):
        self.fitted = False
        self.mean = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = np.mean(X, axis=0)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("DummyTransformer has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        return X - self.mean


class DummyEstimator:
    def __init__(self):
        self.fitted = False
        self.majority_class = None

    def fit(self, X, y):
        self.fitted = True
        values, counts = np.unique(y, return_counts=True)
        self.majority_class = values[np.argmax(counts)]
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("DummyEstimator has not been fitted yet.")
        return np.full(X.shape[0], self.majority_class)


class BadTransformerNoTransform:
    def fit(self, X):
        return self


class BadEstimatorNoPredict:
    def fit(self, X, y=None):
        return self


class BadStepNoFit:
    def transform(self, X):
        return X


def test_pipeline_fit_predict_basic():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])

    pipe = Pipeline([
        ("transform", DummyTransformer()),
        ("model", DummyEstimator())
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (3,)
    assert np.array_equal(preds, np.array([1, 1, 1]))


def test_pipeline_transform_after_fit():
    X = np.array([[1, 2], [3, 4], [5, 6]])

    pipe = Pipeline([
        ("transform", DummyTransformer())
    ])

    pipe.fit(X)
    X_out = pipe.transform(X)

    assert X_out.shape == X.shape
    assert np.allclose(np.mean(X_out, axis=0), [0, 0])


def test_pipeline_fit_transform():
    X = np.array([[1, 2], [3, 4], [5, 6]])

    pipe = Pipeline([
        ("transform", DummyTransformer())
    ])

    X_out = pipe.fit_transform(X)

    assert X_out.shape == X.shape
    assert np.allclose(np.mean(X_out, axis=0), [0, 0])


def test_pipeline_empty_steps():
    with pytest.raises(ValueError):
        Pipeline([])


def test_pipeline_bad_transformer_missing_transform():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    pipe = Pipeline([
        ("bad", BadTransformerNoTransform()),
        ("model", DummyEstimator())
    ])

    with pytest.raises(ValueError):
        pipe.fit(X, y)


def test_pipeline_bad_step_missing_fit():
    X = np.array([[1, 2], [3, 4]])

    pipe = Pipeline([
        ("bad", BadStepNoFit())
    ])

    with pytest.raises(ValueError):
        pipe.fit(X)


def test_pipeline_predict_final_step_missing_predict():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    pipe = Pipeline([
        ("transform", DummyTransformer()),
        ("bad_model", BadEstimatorNoPredict())
    ])

    pipe.fit(X, y)

    with pytest.raises(ValueError):
        pipe.predict(X)


def test_pipeline_predict_before_fit_raises_from_transformer():
    X = np.array([[1, 2], [3, 4]])

    pipe = Pipeline([
        ("transform", DummyTransformer()),
        ("model", DummyEstimator())
    ])

    with pytest.raises(ValueError):
        pipe.predict(X)


def test_pipeline_transform_only_applies_transformers_before_model():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])

    pipe = Pipeline([
        ("transform", DummyTransformer()),
        ("model", DummyEstimator())
    ])

    pipe.fit(X, y)
    X_transformed = pipe.transform(X)

    assert X_transformed.shape == X.shape
    assert np.allclose(np.mean(X_transformed, axis=0), [0, 0])


def test_pipeline_single_estimator_predict():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([2, 2, 1])

    pipe = Pipeline([
        ("model", DummyEstimator())
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert np.array_equal(preds, np.array([2, 2, 2]))