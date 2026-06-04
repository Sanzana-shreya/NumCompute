# testing ColumnTransformer for handling mixed numeric and categorical data
import numpy as np
import pytest

from numcompute.preprocessing import ColumnTransformer


def test_column_transformer_basic():
    # input mixed numeric + categorical data
    X = np.array([
        [25, "Male"],
        [30, "Female"],
        [22, "Female"]
    ], dtype=object)

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
    ], dtype=object)

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
    ], dtype=object)

    ct = ColumnTransformer(
        num_cols=[0],
        cat_cols=[5]  # invalid column index
    )

    # should raise error due to invalid column selection
    with pytest.raises(Exception):
        ct.fit_transform(X)


def test_column_transformer_numeric_only():
    # only numeric columns
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], dtype=object)

    ct = ColumnTransformer(
        num_cols=[0, 1],
        cat_cols=[]
    )

    X_out = ct.fit_transform(X)

    # output should keep same number of rows
    assert X_out.shape[0] == X.shape[0]

    # output should have 2 numeric columns
    assert X_out.shape[1] == 2


def test_column_transformer_categorical_only():
    # only categorical columns
    X = np.array([
        ["Red", "Small"],
        ["Blue", "Medium"],
        ["Red", "Medium"]
    ], dtype=object)

    ct = ColumnTransformer(
        num_cols=[],
        cat_cols=[0, 1]
    )

    X_out = ct.fit_transform(X)

    # output should keep same number of rows
    assert X_out.shape[0] == X.shape[0]

    # encoded output should be numeric
    assert np.issubdtype(X_out.dtype, np.number)


def test_column_transformer_not_fitted():
    # transform before fit
    X = np.array([
        [25, "Male"],
        [30, "Female"]
    ], dtype=object)

    ct = ColumnTransformer(
        num_cols=[0],
        cat_cols=[1]
    )

    # should raise error because transformer is not fitted
    with pytest.raises(ValueError):
        ct.transform(X)


def test_column_transformer_invalid_input_shape():
    # invalid 1D input
    X = np.array([25, "Male"], dtype=object)

    ct = ColumnTransformer(
        num_cols=[0],
        cat_cols=[1]
    )

    # should raise error because input is not 2D
    with pytest.raises(ValueError):
        ct.fit(X)


def test_column_transformer_partial_fit():
    # partial_fit with mixed data
    X1 = np.array([
        [25, "Male"],
        [30, "Female"]
    ], dtype=object)

    X2 = np.array([
        [22, "Female"]
    ], dtype=object)

    ct = ColumnTransformer(
        num_cols=[0],
        cat_cols=[1]
    )

    ct.partial_fit(X1)
    ct.partial_fit(X2)

    X_out = ct.transform(X1)

    # output should exist after streaming fit
    assert X_out is not None

    # number of rows must stay same
    assert X_out.shape[0] == X1.shape[0]