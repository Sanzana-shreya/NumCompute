import numpy as np

# For Basic Statistics 
# Mean

def mean(X, axis=None):
    return np.nanmean(X, axis=axis)

# Median
def median(X, axis=None):
    return np.nanmedian(X, axis=axis)

# Standard Deviation
def std(X, axis=None):
    return np.nanstd(X, axis=axis)

# Minimum
def minimum(X, axis=None):
    return np.nanmin(X, axis=axis)

# Maximum
def maximum(X, axis=None):
    return np.nanmax(X, axis=axis)


# For Histogram 

def histogram(X, bins=10, range=None):
    return np.histogram(X, bins=bins, range=range)


#  For Quantiles 

def quantiles(X, q, axis=None):
    
    # q: list or scalar (0–100)
    
    return np.nanpercentile(X, q, axis=axis)


# Axis-wise behaviour helper

def describe(X, axis=0):
    
    # Returns mean, std, min, max along axis
    
    return {
        "mean": mean(X, axis),
        "std": std(X, axis),
        "min": minimum(X, axis),
        "max": maximum(X, axis),
    }