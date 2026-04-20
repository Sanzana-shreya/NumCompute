import numpy as np
from numcompute.preprocessing import ColumnTransformer

X = np.array([
    [25, "Male"],
    [30, "Female"],
    [22, "Female"]
])

ct = ColumnTransformer(
    num_cols=[0],
    cat_cols=[1]
)

X_out = ct.fit_transform(X)

print("Original:\n", X)
print("\nTransformed:\n", X_out)

