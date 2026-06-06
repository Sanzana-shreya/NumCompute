# testing StandardScaler and MinMaxScaler with simple numeric data
import numpy as np
import pytest

from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler


def test_standard_scaler_basic():
    # simple numeric dataset for testing scaling
    # sample data
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=float)

    # StandardScaler
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X)

    # mean should be ~0 and std ~1 (column-wise)
    assert np.allclose(np.mean(X_std, axis=0), [0, 0])
    assert np.allclose(np.std(X_std, axis=0), [1, 1])


def test_minmax_scaler_basic():
    # simple numeric dataset for testing scaling
    # sample data
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=float)

    # MinMaxScaler
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X)

    # values should be scaled between 0 and 1
    assert np.allclose(np.min(X_minmax, axis=0), [0, 0])
    assert np.allclose(np.max(X_minmax, axis=0), [1, 1])


def test_standard_scaler_transform_consistency():
    # simple numeric dataset for testing scaling
    # sample data
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=float)

    std_scaler = StandardScaler()
    std_scaler.fit(X)

    X1 = std_scaler.transform(X)
    X2 = std_scaler.fit_transform(X)

    assert np.allclose(X1, X2)


def test_minmax_scaler_transform_consistency():
    # simple numeric dataset for testing scaling
    # sample data
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=float)

    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(X)

    X1 = minmax_scaler.transform(X)
    X2 = minmax_scaler.fit_transform(X)

    assert np.allclose(X1, X2)


def test_scaler_non_numeric():
    # Edge case: non-numeric input
    X = np.array([["a", "b"], ["c", "d"]], dtype=object)

    with pytest.raises(ValueError):
        StandardScaler().fit_transform(X)

    with pytest.raises(ValueError):
        MinMaxScaler().fit_transform(X)


def test_standard_scaler_partial_fit():
    # simple numeric dataset for testing streaming scaling
    X1 = np.array([
        [1, 2],
        [3, 4]
    ], dtype=float)

    X2 = np.array([
        [5, 6]
    ], dtype=float)

    scaler = StandardScaler()
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    X_scaled = scaler.transform(X1)

    assert X_scaled.shape == X1.shape


def test_minmax_scaler_partial_fit():
    # simple numeric dataset for testing streaming scaling
    X1 = np.array([
        [1, 2],
        [3, 4]
    ], dtype=float)

    X2 = np.array([
        [5, 6]
    ], dtype=float)

    scaler = MinMaxScaler()
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    X_scaled = scaler.transform(X1)

    assert X_scaled.shape == X1.shape