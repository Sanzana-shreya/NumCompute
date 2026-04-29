import numpy as np
import pytest

from numcompute.stats import mean, median, std, minimum, maximum, quantiles, describe


# For stats.py


def test_normal_case():
    # Normal Case 
    X = np.array([1, 2, 3, 4, 5])

    assert np.isclose(mean(X), 3)
    assert np.isclose(median(X), 3)
    assert np.isclose(std(X), np.std(X))
    assert np.isclose(minimum(X), 1)
    assert np.isclose(maximum(X), 5)
    assert np.allclose(quantiles(X, [25, 50, 75]), [2, 3, 4])


def test_with_nan_values():
    # With NaN values
    X_nan = np.array([1, 2, np.nan, 4, 5])

    assert np.isclose(mean(X_nan), 3)
    assert np.isclose(median(X_nan), 3)
    assert np.isclose(std(X_nan), np.nanstd(X_nan))


def test_axis_case():
    # 2D Array (axis test) 
    X_2d = np.array([[1, 2, 3],
                     [4, 5, 6]])

    assert np.allclose(mean(X_2d, axis=0), [2.5, 3.5, 4.5])
    assert np.allclose(mean(X_2d, axis=1), [2, 5])


def test_empty_array():
    # Edge Case: Empty Array
    with pytest.raises(ValueError):
        mean([])


def test_non_numeric():
    # Edge Case: Non-numeric 
    with pytest.raises(ValueError):
        mean(["a", "b", "c"])


def test_invalid_quantile():
    #  Edge Case: Invalid Quantile 
    X = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        quantiles(X, [-10, 50, 110])


def test_describe():
    # Describe 
    X = np.array([1, 2, 3, 4, 5])
    result = describe(X)

    assert isinstance(result, dict)
    assert np.isclose(result["mean"], 3)
    assert np.isclose(result["std"], np.std(X))
    assert np.isclose(result["min"], 1)
    assert np.isclose(result["max"], 5)