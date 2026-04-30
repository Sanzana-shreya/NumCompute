import numpy as np
from typing import Union


def rank(data: np.ndarray, method: str = 'average') -> np.ndarray:
    """
    Compute ranks of data with tie handling.
    """

<<<<<<< HEAD
=======

def rank(data, method='average'):
    """
    Compute ranks of a 1D array with support for different tie-handling strategies.

    Parameters
    ----------
    data : array-like of shape (n_samples,)
        Input array to rank.
    method : {'average', 'dense', 'ordinal'}, default='average'
        Method used to assign ranks in case of ties:
        - 'average': average rank for tied values
        - 'dense': consecutive ranks without gaps
        - 'ordinal': ranks assigned in order of appearance

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Array of ranks corresponding to input data. NaN values are preserved.

    Raises
    ------
    ValueError
        If method is invalid or input is not 1D.

    Notes
    -----
    - NaN values are excluded from ranking and restored in output.
    - Stable sorting is used to preserve input order for ties.

    Complexity
    ----------
    Time: O(n log n)
    Space: O(n)
    """
>>>>>>> fb914272e45a434296a4db2f75894d599f0f7691
    if method not in ['average', 'dense', 'ordinal']:
        raise ValueError("Invalid method. Supported methods are: 'average', 'dense', 'ordinal'.")

    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError(f"rank expects a 1D array, got shape {data.shape}")

    # Separate NaN values so they are preserved in output
    nan = np.isnan(data)
    valid = data[~nan]

    if len(valid) == 0:
        return np.full_like(data, np.nan, dtype=float)

    # Stable sort ensures deterministic ranking for ordinal method
    order = np.argsort(valid, kind='stable')
    sorted_vals = valid[order]

    ranks = np.empty(len(valid), dtype=float)

    if method == 'ordinal':
        # Rank strictly by sorted order (no tie handling)
        ranks[order] = np.arange(1, len(valid) + 1)

    elif method == 'dense':
        # Each unique value gets a consecutive rank (no gaps)
        unique_vals, inverse = np.unique(sorted_vals, return_inverse=True)
        ranks[order] = inverse + 1

    elif method == 'average':
        # Compute average rank for tied groups
        unique_vals, inverse, counts = np.unique(
            sorted_vals, return_inverse=True, return_counts=True
        )

        # Compute start/end ranks of each tie group
        cumulative = np.cumsum(counts)
        start = cumulative - counts + 1
        end = cumulative

        avg_ranks = (start + end) / 2.0

        ranks[order] = avg_ranks[inverse]

    # Restore NaN positions in final output
    full_ranks = np.full_like(data, np.nan, dtype=float)
    full_ranks[~nan] = ranks

    return full_ranks


<<<<<<< HEAD
def percentile(
    data: np.ndarray,
    q: float,
    interpolation: str = 'linear'
) -> Union[float, np.floating]:
    """
    Compute the q-th percentile of the data.
    """

=======
def percentile(data, q, interpolation='linear'):
    """
    Compute the q-th percentile of a 1D array with configurable interpolation.

    Parameters
    ----------
    data : array-like of shape (n_samples,)
        Input array.
    q : float
        Percentile to compute, must be in the range [0, 100].
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, default='linear'
        Interpolation method:
        - 'linear': linear interpolation between values
        - 'lower': lower bound
        - 'higher': upper bound
        - 'midpoint': average of lower and upper

    Returns
    -------
    float
        Computed percentile value. Returns NaN if all values are NaN.

    Raises
    ------
    ValueError
        If interpolation method is invalid, input is not 1D, or q is out of bounds.

    Notes
    -----
    - NaN values are ignored before computation.
    - Behavior matches NumPy-style percentile logic.

    Complexity
    ----------
    Time: O(n log n)
    Space: O(n)
    """
>>>>>>> fb914272e45a434296a4db2f75894d599f0f7691
    if interpolation not in ['linear', 'lower', 'higher', 'midpoint']:
        raise ValueError(
            "Invalid interpolation method. Supported methods are: "
            "'linear', 'lower', 'higher', 'midpoint'."
        )

    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError(f"percentile expects a 1D array, got shape {data.shape}")

    if not (0 <= q <= 100):
        raise ValueError("q must be between 0 and 100")

<<<<<<< HEAD
    # Remove NaNs before computation
=======
>>>>>>> fb914272e45a434296a4db2f75894d599f0f7691
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return np.nan

    # Sort for percentile computation
    data = np.sort(data)

    # Position in sorted array
    pos = (q / 100) * (len(data) - 1)
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))

    if interpolation == 'lower':
        return float(data[lower])

    elif interpolation == 'higher':
        return float(data[upper])

    elif interpolation == 'midpoint':
        return float((data[lower] + data[upper]) / 2.0)

    elif interpolation == 'linear':
        # Exact index case
        if lower == upper:
            return float(data[lower])

        # Linear interpolation between two points
        weight = pos - lower
<<<<<<< HEAD
        return float(data[lower] * (1 - weight) + data[upper] * weight)
=======
        return data[lower] * (1 - weight) + data[upper] * weight
>>>>>>> fb914272e45a434296a4db2f75894d599f0f7691
