#testing StandardScaler and MinMaxScaler with simple numeric data
import numpy as np
from numcompute.preprocessing import StandardScaler, MinMaxScaler

#simple numeric dataset for testing scaling
# sample data
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

# StandardScaler
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)

print("Original:\n", X)
print("\nStandard Scaled:\n", X_std)

# MinMaxScaler
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)

print("\nMinMax Scaled:\n", X_minmax)