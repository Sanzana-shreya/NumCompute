import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numcompute.pipeline import Pipeline


class DummyStep:
    def __init__(self):
        self.factor = 2

    def fit(self, X):
        return X

    def transform(self, X):
        return [x * self.factor for x in X]


steps = [
    ("step1", DummyStep()),
    ("step2", DummyStep())
]

pipeline = Pipeline(steps)

X = [1, 2, 3]

pipeline.fit(X)
result = pipeline.predict(X)

print(result)