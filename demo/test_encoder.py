# testing OneHotEncoder with simple categorical data
import numpy as np
import pytest

from numcompute.preprocessing import OneHotEncoder


def test_onehot_encoder_basic():
    # sample categorical data
    X = np.array([
        ["Red", "S"],
        ["Blue", "M"],
        ["Red", "M"],
        ["Green", "S"]
    ], dtype=object)

    encoder = OneHotEncoder()

    # Fit and transform data
    X_encoded = encoder.fit_transform(X)

    # Check output exists
    assert X_encoded is not None

    # Check output is numpy array
    assert isinstance(X_encoded, np.ndarray)

    # Check number of rows stays same
    assert X_encoded.shape[0] == X.shape[0]

    # Encoded output should be numeric
    assert np.issubdtype(X_encoded.dtype, np.number)


def test_onehot_encoder_shape_expansion():
    # sample categorical data
    X = np.array([
        ["Red", "S"],
        ["Blue", "M"]
    ], dtype=object)

    encoder = OneHotEncoder()

    X_encoded = encoder.fit_transform(X)

    # Encoding should increase number of columns
    assert X_encoded.shape[1] > X.shape[1]


def test_onehot_encoder_invalid_input():
    # Invalid input (1D input)
    X = np.array(["Red", "Blue"], dtype=object)

    encoder = OneHotEncoder()

    # Should raise error for invalid input shape
    with pytest.raises(ValueError):
        encoder.fit_transform(X)


def test_onehot_encoder_transform_before_fit():
    # Transform before fit
    X = np.array([
        ["Red", "S"],
        ["Blue", "M"]
    ], dtype=object)

    encoder = OneHotEncoder()

    # Should raise error because encoder is not fitted
    with pytest.raises(ValueError):
        encoder.transform(X)


def test_onehot_encoder_partial_fit():
    # partial_fit with categorical data
    X1 = np.array([
        ["Red", "S"],
        ["Blue", "M"]
    ], dtype=object)

    X2 = np.array([
        ["Green", "L"]
    ], dtype=object)

    encoder = OneHotEncoder()

    # Incremental fit
    encoder.partial_fit(X1)
    encoder.partial_fit(X2)

    X_encoded = encoder.transform(X1)

    # Check output exists
    assert X_encoded is not None

    # Check output is numeric
    assert np.issubdtype(X_encoded.dtype, np.number)


def test_onehot_encoder_column_mismatch():
    # Different number of columns at transform time
    X_train = np.array([
        ["Red", "S"],
        ["Blue", "M"]
    ], dtype=object)

    X_test = np.array([
        ["Red"]
    ], dtype=object)

    encoder = OneHotEncoder()
    encoder.fit(X_train)

    # Should raise error because number of columns differs
    with pytest.raises(ValueError):
        encoder.transform(X_test)