#testing pipeline with preprocessing on mixed data
import numpy as np
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute.preprocessing import OneHotEncoder

X = np.array([
    [25, "Male"],
    [30, "Female"],
    [22, "Female"]
])

#pipeline applying encoding step
pipe = Pipeline([
    ("encode", OneHotEncoder())
])

X_out = pipe.fit_transform(X)

print("Original:\n", X)
print("\nPipeline Output:\n", X_out) 
