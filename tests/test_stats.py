import numpy as np
import pytest

from numcompute.stats import (
    mean,
    median,
    std,
    minimum,
    maximum,
    variance,
    histogram,
    quantiles,
    describe,
)


def test_mean_normal_case():
    X = np.array([1, 2, 3, 4, 5])
    assert np.isclose(mean(X), 3.0)


def test_median_normal_case():
    X = np.array([1, 2, 3, 4, 5])
    assert np.isclose(median(X), 3.0)


def test_std_normal_case():
    X = np.array([1, 2, 3, 4, 5])
    assert np.isclose(std(X), np.std(X))


def test_variance_normal_case():
    X = np.array([1, 2, 3, 4, 5])
    assert np.isclose(variance(X), np.var(X))


def test_minimum_maximum_normal_case():
    X = np.array([1, 2, 3, 4, 5])
    assert minimum(X) == 1
    assert maximum(X) == 5


def test_stats_ignore_nan_values():
    X = np.array([1, 2, np.nan, 4, 5])

    assert np.isclose(mean(X), np.nanmean(X))
    assert np.isclose(median(X), np.nanmedian(X))
    assert np.isclose(std(X), np.nanstd(X))
    assert np.isclose(variance(X), np.nanvar(X))


def test_axis_wise_mean_axis_0():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    result = mean(X, axis=0)

    assert np.allclose(result, np.array([2.5, 3.5, 4.5]))


def test_axis_wise_mean_axis_1():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    result = mean(X, axis=1)

    assert np.allclose(result, np.array([2.0, 5.0]))


def test_axis_wise_std():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    result = std(X, axis=0)

    assert np.allclose(result, np.std(X, axis=0))


def test_quantiles_normal_case():
    X = np.array([1, 2, 3, 4, 5])

    result = quantiles(X, [25, 50, 75])

    assert np.allclose(result, np.array([2, 3, 4]))


def test_quantiles_ignore_nan_values():
    X = np.array([1, 2, np.nan, 4, 5])

    result = quantiles(X, [25, 50, 75])
    expected = np.nanpercentile(X, [25, 50, 75])

    assert np.allclose(result, expected)


def test_quantiles_invalid_low_value():
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        quantiles(X, -10)


def test_quantiles_invalid_high_value():
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        quantiles(X, 110)


def test_histogram_basic():
    X = np.array([1, 2, 3, 4, 5])

    counts, bins = histogram(X, bins=2)

    assert counts.sum() == len(X)
    assert len(bins) == 3


def test_histogram_invalid_bins():
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        histogram(X, bins=0)


def test_empty_array_raises_error():
    with pytest.raises(ValueError):
        mean([])


def test_non_numeric_input_raises_error():
    with pytest.raises(ValueError):
        mean(["a", "b", "c"])


def test_none_input_raises_error():
    with pytest.raises(ValueError):
        mean(None)


def test_describe_contains_expected_keys():
    X = np.array([1, 2, 3, 4, 5])

    result = describe(X)

    expected_keys = {"mean", "median", "variance", "std", "min", "max"}

    assert set(result.keys()) == expected_keys


def test_describe_values_are_correct():
    X = np.array([1, 2, 3, 4, 5])

    result = describe(X)

    assert np.isclose(result["mean"], 3.0)
    assert np.isclose(result["median"], 3.0)
    assert np.isclose(result["variance"], np.var(X))
    assert np.isclose(result["std"], np.std(X))
    assert result["min"] == 1
    assert result["max"] == 5


import warnings

def test_all_nan_array_returns_nan():
    X = np.array([np.nan, np.nan])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        assert np.isnan(mean(X))
        assert np.isnan(median(X))
        assert np.isnan(std(X))
        assert np.isnan(variance(X))