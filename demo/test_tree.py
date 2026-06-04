import numpy as np
import pytest

from numcompute.tree import StreamingDecisionTreeClassifier


# For tree.py


def test_tree_normal_case():
    # Normal case
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = StreamingDecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == y.shape


def test_tree_partial_fit_case():
    # Streaming partial_fit case
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = StreamingDecisionTreeClassifier(max_depth=3)
    model.partial_fit(X[:3], y[:3], classes=np.array([0, 1]))
    model.partial_fit(X[3:], y[3:])

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_tree_predict_proba_shape():
    # Predict probability shape
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = StreamingDecisionTreeClassifier()
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (4, 2)


def test_tree_with_nan_values():
    # With NaN values
    X = np.array([[0], [1], [np.nan], [4]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = StreamingDecisionTreeClassifier()
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_tree_entropy_case():
    # Entropy criterion
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = StreamingDecisionTreeClassifier(criterion="entropy")
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_tree_predict_before_fit():
    # Predict before fit
    X = np.array([[0], [1]], dtype=float)

    model = StreamingDecisionTreeClassifier()

    with pytest.raises(ValueError):
        model.predict(X)


def test_tree_invalid_X():
    # Invalid X shape
    X = np.array([1, 2, 3])
    y = np.array([0, 1, 0])

    model = StreamingDecisionTreeClassifier()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_tree_invalid_y():
    # Invalid y shape
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([[0], [1], [0]])

    model = StreamingDecisionTreeClassifier()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_tree_shape_mismatch():
    # Shape mismatch
    X = np.array([[1], [2]], dtype=float)
    y = np.array([0])

    model = StreamingDecisionTreeClassifier()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_tree_invalid_criterion():
    # Invalid criterion
    X = np.array([[0], [1], [2]], dtype=float)
    y = np.array([0, 1, 1])

    model = StreamingDecisionTreeClassifier(criterion="wrong")

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_tree_feature_mismatch_predict():
    # Feature mismatch in predict
    X = np.array([[0], [1], [2]], dtype=float)
    y = np.array([0, 1, 1])

    model = StreamingDecisionTreeClassifier()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1, 2]], dtype=float))


def test_tree_feature_mismatch_partial_fit():
    # Feature mismatch in partial_fit
    X = np.array([[0], [1], [2]], dtype=float)
    y = np.array([0, 1, 1])

    model = StreamingDecisionTreeClassifier()
    model.partial_fit(X, y, classes=np.array([0, 1]))

    with pytest.raises(ValueError):
        model.partial_fit(np.array([[1, 2]], dtype=float), np.array([1]))


def test_tree_single_class_case():
    # Single class
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([1, 1, 1])

    model = StreamingDecisionTreeClassifier()
    model.fit(X, y)

    y_pred = model.predict(X)
    assert np.all(y_pred == 1)  