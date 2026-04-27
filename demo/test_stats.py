import numpy as np
from numcompute.stats import mean, median, std, minimum, maximum, quantiles, describe

# For stats.py

print("\n===== STAT TESTS =====")

# Normal Case 

X = np.array([1, 2, 3, 4, 5])
print("Mean:", mean(X))                  # 3
print("Median:", median(X))              # 3
print("Std:", std(X))                   # ~1.414
print("Min:", minimum(X))               # 1
print("Max:", maximum(X))               # 5
print("Quantiles:", quantiles(X, [25, 50, 75]))


# With NaN values

X_nan = np.array([1, 2, np.nan, 4, 5])
print("\nWith NaN:")
print("Mean:", mean(X_nan))             # ignores NaN
print("Median:", median(X_nan))
print("Std:", std(X_nan))


# 2D Array (axis test) 

X_2d = np.array([[1, 2, 3],
                 [4, 5, 6]])
print("\nAxis Test:")
print("Mean axis=0:", mean(X_2d, axis=0))
print("Mean axis=1:", mean(X_2d, axis=1))


# Edge Case: Empty Array

try:
    mean([])
except ValueError as e:
    print("\nEmpty Array Error:", e)


# Edge Case: Non-numeric 

try:
    mean(["a", "b", "c"])
except ValueError as e:
    print("Non-numeric Error:", e)


#  Edge Case: Invalid Quantile 

try:
    quantiles(X, [-10, 50, 110])
except ValueError as e:
    print("Invalid Quantile Error:", e)


# Describe 

print("\nDescribe:", describe(X))



