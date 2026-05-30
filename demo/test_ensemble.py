import numpy as np
import pytest

from numcompute.ensemble import StreamingRandomForestClassifier


# For ensemble.py


def test_rf_normal_case():
    # Normal case
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = StreamingRandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_rf_partial_fit_case():
    # Streaming partial fit
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = StreamingRandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
    model.partial_fit(X[:3], y[:3], classes=np.array([0, 1]))
    model.partial_fit(X[3:], y[3:])

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_rf_predict_proba_shape():
    # Predict probability shape
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = StreamingRandomForestClassifier(n_estimators=3, random_state=42)
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (4, 2)


def test_rf_predict_before_fit():
    # Predict before fit
    X = np.array([[0], [1]], dtype=float)

    model = StreamingRandomForestClassifier()

    with pytest.raises(ValueError):
        model.predict(X)


def test_rf_invalid_X():
    # Invalid X
    X = np.array([1, 2, 3])
    y = np.array([0, 1, 0])

    model = StreamingRandomForestClassifier()

    with pytest.raises(ValueError):
        model.partial_fit(X, y)


def test_rf_invalid_y():
    # Invalid y
    X = np.array([[1], [2]], dtype=float)
    y = np.array([[0], [1]])

    model = StreamingRandomForestClassifier()

    with pytest.raises(ValueError):
        model.partial_fit(X, y)


def test_rf_shape_mismatch():
    # Shape mismatch
    X = np.array([[1], [2]], dtype=float)
    y = np.array([0])

    model = StreamingRandomForestClassifier()

    with pytest.raises(ValueError):
        model.partial_fit(X, y)


def test_rf_feature_mismatch():
    # Feature mismatch
    X = np.array([[0], [1], [2]], dtype=float)
    y = np.array([0, 1, 1])

    model = StreamingRandomForestClassifier(random_state=42)
    model.partial_fit(X, y, classes=np.array([0, 1]))

    with pytest.raises(ValueError):
        model.predict(np.array([[1, 2]], dtype=float))


def test_rf_without_bootstrap():
    # Without bootstrap
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = StreamingRandomForestClassifier(
        n_estimators=3,
        bootstrap=False,
        random_state=42
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape