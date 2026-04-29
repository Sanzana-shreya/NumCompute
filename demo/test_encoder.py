#testing OneHotEmcoder with simple categorical data
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
    ])

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
    ])

    encoder = OneHotEncoder()

    X_encoded = encoder.fit_transform(X)

    # Encoding should increase number of columns
    assert X_encoded.shape[1] > X.shape[1]


def test_onehot_encoder_invalid_input():
    # Invalid input (non-string mixed type)
    X = np.array([
        [1, 2],
        [3, 4]
    ])

    encoder = OneHotEncoder()

    # Should raise error for non-categorical data
    with pytest.raises(Exception):
        encoder.fit_transform(X)