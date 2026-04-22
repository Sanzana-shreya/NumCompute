import numpy as np
from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, ColumnTransformer

X_num = np.array([
    [1, 2],
    [3, 4]
])

X_cat = np.array([
    ["Red"],
    ["Blue"]
])

X_mixed = np.array([
    [25, "Male"],
    [30, "Female"]
])

scaler = StandardScaler()

try:
    scaler.transform(X_num)
except ValueError as e:
    print("StandardScaler error:", e)


minmax = MinMaxScaler()

try:
    minmax.transform(X_num)
except ValueError as e:
    print("MinMaxScaler error:", e)


encoder = OneHotEncoder()

try:
    encoder.transform(X_cat)
except ValueError as e:
    print("OneHotEncoder error:", e)



ct = ColumnTransformer(num_cols=[0], cat_cols=[1])

try:
    ct.transform(X_mixed)
except ValueError as e:
    print("ColumnTransformer error:", e)

    
