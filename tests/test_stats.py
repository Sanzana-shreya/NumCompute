import numpy as np
import pytest

from numcompute.stats import (
    mean, median, std, variance,
    minimum, maximum, histogram,
    quantiles, describe
)


# For stats.py


def test_normal_case():
    # Normal Case
    X = np.array([1, 2, 3, 4, 5], dtype=float)

    assert np.isclose(mean(X), 3)
    assert np.isclose(median(X), 3)
    assert np.isclose(std(X), np.std(X))
    assert np.isclose(variance(X), np.var(X))
    assert np.isclose(minimum(X), 1)
    assert np.isclose(maximum(X), 5)
    assert np.allclose(quantiles(X, [25, 50, 75]), [2, 3, 4])


def test_with_nan_values():
    # With NaN values
    X_nan = np.array([1, 2, np.nan, 4, 5], dtype=float)

    assert np.isclose(mean(X_nan), 3)
    assert np.isclose(median(X_nan), 3)
    assert np.isclose(std(X_nan), np.nanstd(X_nan))
    assert np.isclose(variance(X_nan), np.nanvar(X_nan))


def test_axis_case():
    # 2D Array (axis test)
    X_2d = np.array([[1, 2, 3],
                     [4, 5, 6]], dtype=float)

    assert np.allclose(mean(X_2d, axis=0), [2.5, 3.5, 4.5])
    assert np.allclose(mean(X_2d, axis=1), [2, 5])


def test_invalid_axis():
    # Invalid axis
    X = np.array([1, 2, 3], dtype=float)

    with pytest.raises(ValueError):
        mean(X, axis=1)


def test_empty_array():
    # Edge Case: Empty Array
    with pytest.raises(ValueError):
        mean([])


def test_non_numeric():
    # Edge Case: Non-numeric
    with pytest.raises(ValueError):
        mean(["a", "b", "c"])


def test_invalid_quantile():
    # Edge Case: Invalid Quantile
    X = np.array([1, 2, 3, 4, 5], dtype=float)

    with pytest.raises(ValueError):
        quantiles(X, [-10, 50, 110])


def test_histogram_case():
    # Histogram
    X = np.array([1, 2, 2, 3, 4], dtype=float)

    hist, bin_edges = histogram(X, bins=3)

    assert len(hist) == 3
    assert len(bin_edges) == 4


def test_invalid_histogram_bins():
    # Invalid histogram bins
    X = np.array([1, 2, 3], dtype=float)

    with pytest.raises(ValueError):
        histogram(X, bins=0)


def test_invalid_histogram_range():
    # Invalid histogram range
    X = np.array([1, 2, 3], dtype=float)

    with pytest.raises(ValueError):
        histogram(X, bins=3, range=(5, 1))


def test_describe():
    # Describe
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    result = describe(X)

    assert isinstance(result, dict)
    assert np.isclose(result["mean"], 3)
    assert np.isclose(result["median"], 3)
    assert np.isclose(result["variance"], np.var(X))
    assert np.isclose(result["std"], np.std(X))
    assert np.isclose(result["min"], 1)
    assert np.isclose(result["max"], 5)