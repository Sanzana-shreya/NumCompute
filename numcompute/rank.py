import numpy as np

def rank(data, method='average'):
    if method not in ['average', 'dense', 'ordinal']:
        raise ValueError("Invalid method. Supported methods are: 'average', 'dense', 'ordinal'.")

    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError("rank expects a 1D array")

    # Handle NaNs separately
    nan = np.isnan(data)
    valid = data[~nan]

    if len(valid) == 0:
        return np.full_like(data, np.nan, dtype=float)

    order = np.argsort(valid, kind='stable')
    sorted_vals = valid[order]

    ranks = np.empty(len(valid), dtype=float)

    if method == 'ordinal':
        ranks[order] = np.arange(1, len(valid) + 1)

    elif method == 'dense':
        unique_vals, inverse = np.unique(sorted_vals, return_inverse=True)
        ranks[order] = inverse + 1

    elif method == 'average':
        unique_vals, inverse, counts = np.unique(
            sorted_vals, return_inverse=True, return_counts=True
        )

        cumulative = np.cumsum(counts)
        start = cumulative - counts + 1
        end = cumulative

        avg_ranks = (start + end) / 2.0

        ranks[order] = avg_ranks[inverse]

    full_ranks = np.full_like(data, np.nan, dtype=float)
    full_ranks[~nan] = ranks

    return full_ranks


def percentile(data, q, interpolation='linear'):
    if interpolation not in ['linear', 'lower', 'higher', 'midpoint']:
        raise ValueError("Invalid interpolation method. Supported methods are: 'linear', 'lower', 'higher', 'midpoint'.")

    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError("percentile expects a 1D array")
    if not (0 <= q <= 100):
        raise ValueError("q must be between 0 and 100")
    
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return np.nan

    data = np.sort(data)

    # Position
    pos = (q / 100) * (len(data) - 1)
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))

    if interpolation == 'lower':
        return data[lower]

    elif interpolation == 'higher':
        return data[upper]

    elif interpolation == 'midpoint':
        return (data[lower] + data[upper]) / 2.0

    elif interpolation == 'linear':
        if lower == upper:
            return data[lower]
        weight = pos - lower
        return data[lower] * (1 - weight) + data[upper] * weight