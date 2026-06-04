# testing pipeline with preprocessing on mixed data
import numpy as np
import pytest

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import OneHotEncoder


def test_pipeline_with_onehot_encoder():
    # Input mixed data (numeric + categorical)
    X = np.array([
        [25, "Male"],
        [30, "Female"],
        [22, "Female"]
    ], dtype=object)

    # pipeline applying encoding step
    pipe = Pipeline([
        ("encode", OneHotEncoder())
    ])

    # Apply fit_transform
    X_out = pipe.fit_transform(X)

    # Check output is not None and has expected number of rows
    assert X_out is not None
    assert len(X_out) == len(X)

    # Output should be numeric after encoding
    assert np.issubdtype(np.asarray(X_out).dtype, np.number)


def test_pipeline_with_encoder_without_fit():
    # Input mixed data
    X = np.array([
        [25, "Male"],
        [30, "Female"]
    ], dtype=object)

    pipe = Pipeline([
        ("encode", OneHotEncoder())
    ])

    # Expect error if transform is called before fit
    with pytest.raises(Exception):
        pipe.predict(X)