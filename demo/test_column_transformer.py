#testing ColumnTransformer for handling mixed numeric and categorical data
import numpy as np
import pytest

from numcompute.preprocessing import ColumnTransformer


def test_column_transformer_basic():
    # input mixed numeric + categorical data
    X = np.array([
        [25, "Male"],
        [30, "Female"],
        [22, "Female"]
    ])

    # define column transformer
    ct = ColumnTransformer(
        num_cols=[0],
        cat_cols=[1]
    )

    # fit and transform data
    X_out = ct.fit_transform(X)

    # check output exists
    assert X_out is not None

    # check output type
    assert isinstance(X_out, np.ndarray)

    # number of rows must stay same
    assert X_out.shape[0] == X.shape[0]

    # output should be numeric after transformation
    assert np.issubdtype(X_out.dtype, np.number)


def test_column_transformer_shape_change():
    # input mixed data
    X = np.array([
        [25, "Male"],
        [30, "Female"]
    ])

    ct = ColumnTransformer(
        num_cols=[0],
        cat_cols=[1]
    )

    X_out = ct.fit_transform(X)

    # transformed data should have more columns due to encoding
    assert X_out.shape[1] >= X.shape[1]


def test_column_transformer_invalid_columns():
    # invalid column index
    X = np.array([
        [25, "Male"],
        [30, "Female"]
    ])

    ct = ColumnTransformer(
        num_cols=[0],
        cat_cols=[5]  # invalid column index
    )

    # should raise error due to invalid column selection
    with pytest.raises(Exception):
        ct.fit_transform(X)