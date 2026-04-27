#testqing separate preprocessing for numeric and categorical data
import numpy as np
from numcompute.preprocessing import StandardScaler, OneHotEncoder

X_num = np.array([
    [25],
    [30],
    [22]
])

X_cat = np.array([
    ["Male"],
    ["Female"],
    ["Female"]
])

scaler = StandardScaler()
encoder = OneHotEncoder()

X_num_scaled = scaler.fit_transform(X_num)
X_cat_encoded = encoder.fit_transform(X_cat)

#combining scaled numeroc and encoded categorical data
X_final = np.hstack((X_num_scaled, X_cat_encoded))

print("Scaled numeric:\n", X_num_scaled)
print("\nEncoded categorical:\n", X_cat_encoded)
print("\nFinal combined output:\n", X_final)

