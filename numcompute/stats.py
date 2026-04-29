from typing import Union, Sequence, Optional, Dict, Any
import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]  # Accepts Python lists, tuples, or NumPy arrays of numbers


# Validation Helper

def _validate_array(
    X: ArrayLike,
    allow_empty: bool = False
) -> np.ndarray:
    """
    Validates and converts input into a NumPy array.

    - Ensures input is not None
    - Ensures input is numeric
    - Optionally allows empty arrays
    """
    if X is None:
        raise ValueError("Input cannot be None")

    arr = np.asarray(X)  # Convert input to NumPy array

    if not np.issubdtype(arr.dtype, np.number):  # Checks if array contains numeric data
        raise ValueError("Input must be numeric")

    if not allow_empty and arr.size == 0:   # Prevent empty arrays unless explicitly allowed
        raise ValueError("Input array cannot be empty")

    return arr


# Basic Statistics

# Mean
def mean(X: ArrayLike, axis: Optional[int] = None) -> float:  # Compute mean while ignoring NaN values
    X = _validate_array(X)
    return np.nanmean(X, axis=axis)


# Median
def median(X: ArrayLike, axis: Optional[int] = None) -> float:  # Compute median while ignoring NaN values
    X = _validate_array(X)
    return np.nanmedian(X, axis=axis)


# Standard Deviation
def std(X: ArrayLike, axis: Optional[int] = None) -> float:  # Compute standard deviation while ignoring NaNs
    X = _validate_array(X)
    return np.nanstd(X, axis=axis)


# Minimum
def minimum(X: ArrayLike, axis: Optional[int] = None) -> float:  # Compute minimum value while ignoring NaNs
    X = _validate_array(X)
    return np.nanmin(X, axis=axis)


# Maximum
def maximum(X: ArrayLike, axis: Optional[int] = None) -> float:  # Compute maximum value while ignoring NaNs
    X = _validate_array(X)
    return np.nanmax(X, axis=axis)

# Variance
def variance(
    X: ArrayLike,
    axis: Optional[int] = None
) -> float:
    """
    Compute variance of the data while ignoring NaN values.

    - axis: axis along which variance is computed
    """
    X = _validate_array(X)
    return np.nanvar(X, axis=axis)


# Histogram 
def histogram(
    X: ArrayLike,
    bins: int = 10,
    range: Optional[tuple] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of the data.

    - bins: number of bins (must be positive integer)
    - range: lower and upper range of bins
    """
    X = _validate_array(X, allow_empty=True)  # Allows empty arrays (NumPy can handle this case)

    if not isinstance(bins, int) or bins <= 0:  # Validate bins input
        raise ValueError("bins must be a positive integer")

    return np.histogram(X, bins=bins, range=range)


# Quantiles
def quantiles(
    X: ArrayLike,
    q: Union[float, Sequence[float]],
    axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Compute percentiles (0–100 scale).

    - q can be a single value or a list of percentiles
    - NaN values are ignored
    """
    X = _validate_array(X)

    q_arr = np.asarray(q)  # Convert q into NumPy array for uniform handling

    if np.any((q_arr < 0) | (q_arr > 100)):  # Ensures if all percentile values are within valid range
        raise ValueError("q must be between 0 and 100")

    return np.nanpercentile(X, q_arr, axis=axis)


# Describe
def describe(
    X: ArrayLike,
    axis: int = 0
) -> Dict[str, Any]:
    """
    Return a summary of basic statistics:
    mean, standard deviation, min, max.
    """
    X = _validate_array(X)

    return {
        "mean": mean(X, axis),
        "median": median(X, axis),
        "variance": variance(X, axis),
        "std": std(X, axis),
        "min": minimum(X, axis),
        "max": maximum(X, axis),
    }