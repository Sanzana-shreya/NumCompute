from typing import Union, Sequence, Optional, Dict, Any
import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]  # Accepts lists, tuples, or NumPy arrays


# Validation Helper
def _validate_array(
    X: ArrayLike,
    allow_empty: bool = False
) -> np.ndarray:
    """
    Validates and converts input into a NumPy array.

    Ensures:
    - input is not None
    - input is numeric
    - optionally allows empty arrays
    """
    if X is None:
        raise ValueError("Input cannot be None")

    arr = np.asarray(X)

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("Input must be numeric")

    # Enforce non-empty constraint unless explicitly allowed
    if not allow_empty and arr.size == 0:
        raise ValueError("Input array cannot be empty")

    return arr


# Basic Statistics

# Mean
def mean(X: ArrayLike, axis: Optional[int] = None) -> float:
    # Uses NaN-safe mean to handle missing values in datasets
    X = _validate_array(X)
    return np.nanmean(X, axis=axis)


# Median
def median(X: ArrayLike, axis: Optional[int] = None) -> float:
    # Robust to outliers; ignores NaN values
    X = _validate_array(X)
    return np.nanmedian(X, axis=axis)


# Standard Deviation
def std(X: ArrayLike, axis: Optional[int] = None) -> float:
    # Measures spread; NaN-safe for real-world datasets
    X = _validate_array(X)
    return np.nanstd(X, axis=axis)


# Minimum
def minimum(X: ArrayLike, axis: Optional[int] = None) -> float:
    # Returns smallest value while ignoring missing data
    X = _validate_array(X)
    return np.nanmin(X, axis=axis)


# Maximum
def maximum(X: ArrayLike, axis: Optional[int] = None) -> float:
    # Returns largest value while ignoring missing data
    X = _validate_array(X)
    return np.nanmax(X, axis=axis)


# Histogram 
def histogram(
    X: ArrayLike,
    bins: int = 10,
    range: Optional[tuple] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of the data.

    Parameters:
    - bins: number of intervals (must be positive integer)
    - range: optional min/max range for binning
    """
    X = _validate_array(X, allow_empty=True)

    # Ensure valid bin configuration for NumPy histogram
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")

    return np.histogram(X, bins=bins, range=range)


# Quantiles
def quantiles(
    X: ArrayLike,
    q: Union[float, Sequence[float]],
    axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Compute quantiles (percentile scale: 0–100).

    q can be scalar or list of values.
    NaN values are ignored for robustness in real datasets.
    """
    X = _validate_array(X)

    q_arr = np.asarray(q)

    # Ensure percentile range is valid
    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be between 0 and 100")

    return np.nanpercentile(X, q_arr, axis=axis)


# Describe
def describe(
    X: ArrayLike,
    axis: int = 0
) -> Dict[str, Any]:
    """
    Returns summary statistics of dataset.

    Includes:
    - mean
    - standard deviation
    - min
    - max
    """
    X = _validate_array(X)

    return {
        "mean": mean(X, axis),
        "std": std(X, axis),
        "min": minimum(X, axis),
        "max": maximum(X, axis),
    }