#testing OneHotEmcoder with simple categorical data
import numpy as np
from numcompute.preprocessing import OneHotEncoder

# sample categorical data
X = np.array([
    ["Red", "S"],
    ["Blue", "M"],
    ["Red", "M"],
    ["Green", "S"]
])

encoder = OneHotEncoder()

X_encoded = encoder.fit_transform(X)

print("Original:\n", X)
print("\nEncoded:\n", X_encoded)