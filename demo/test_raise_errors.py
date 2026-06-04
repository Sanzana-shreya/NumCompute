import numpy as np
import pytest

from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, ColumnTransformer


def test_standard_scaler_transform_before_fit():
    X_num = np.array([
        [1, 2],
        [3, 4]
    ], dtype=float)

    scaler = StandardScaler()

    # StandardScaler
    with pytest.raises(ValueError):
        scaler.transform(X_num)


def test_minmax_scaler_transform_before_fit():
    X_num = np.array([
        [1, 2],
        [3, 4]
    ], dtype=float)

    minmax = MinMaxScaler()

    # MinMaxScaler
    with pytest.raises(ValueError):
        minmax.transform(X_num)


def test_onehot_encoder_transform_before_fit():
    X_cat = np.array([
        ["Red"],
        ["Blue"]
    ], dtype=object)

    encoder = OneHotEncoder()

    # OneHotEncoder
    with pytest.raises(ValueError):
        encoder.transform(X_cat)


def test_column_transformer_transform_before_fit():
    X_mixed = np.array([
        [25, "Male"],
        [30, "Female"]
    ], dtype=object)

    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])

    # ColumnTransformer
    with pytest.raises(ValueError):
        ct.transform(X_mixed)