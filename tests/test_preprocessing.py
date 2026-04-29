import numpy as np
import pytest

from numcompute.preprocessing import (
    Imputer,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    ColumnTransformer,
)


def test_standard_scaler_basic():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), [0, 0])
    assert np.allclose(np.std(X_scaled, axis=0), [1, 1])


def test_standard_scaler_constant_column():
    X = np.array([[5], [5], [5]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled, [[0], [0], [0]])


def test_standard_scaler_transform_before_fit():
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.transform(np.array([[1, 2]]))


def test_minmax_scaler_basic():
    X = np.array([[1], [3], [5]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled, [[0], [0.5], [1]])


def test_minmax_scaler_constant_column():
    X = np.array([[5], [5], [5]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled, [[0], [0], [0]])


def test_simple_imputer_mean():
    X = np.array([[1.0], [np.nan], [3.0]])
    imputer = Imputer(strategy="mean")
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out, [[1.0], [2.0], [3.0]])


def test_simple_imputer_median():
    X = np.array([[1.0], [np.nan], [5.0]])
    imputer = Imputer(strategy="median")
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out, [[1.0], [3.0], [5.0]])


def test_simple_imputer_constant():
    X = np.array([[1.0], [np.nan], [5.0]])
    imputer = Imputer(strategy="constant", fill_value=-1)
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out, [[1.0], [-1.0], [5.0]])


def test_simple_imputer_invalid_strategy():
    with pytest.raises(ValueError):
        Imputer(strategy="mode")


def test_onehot_encoder_basic():
    X = np.array([
        ["Red", "S"],
        ["Blue", "M"],
        ["Red", "M"],
        ["Green", "S"]
    ])

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    assert X_encoded.shape == (4, 5)
    assert np.all((X_encoded == 0) | (X_encoded == 1))


def test_onehot_encoder_transform_before_fit():
    encoder = OneHotEncoder()

    with pytest.raises(ValueError):
        encoder.transform(np.array([["Red"]]))


def test_column_transformer_mixed_data():
    X = np.array([
        [25, "Male"],
        [30, "Female"],
        [22, "Female"]
    ], dtype=object)

    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])
    X_out = ct.fit_transform(X)

    assert X_out.shape == (3, 3)


def test_column_transformer_with_nan_numeric():
    X = np.array([
        [25, "Male"],
        [np.nan, "Female"],
        [35, "Female"]
    ], dtype=object)

    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])
    X_out = ct.fit_transform(X)

    assert np.all(np.isfinite(X_out.astype(float)))


def test_column_transformer_no_columns_selected():
    X = np.array([[1, "A"], [2, "B"]], dtype=object)
    ct = ColumnTransformer(num_cols=[], cat_cols=[])

    with pytest.raises(ValueError):
        ct.fit_transform(X)


def test_wrong_dimension_input():
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.fit(np.array([1, 2, 3]))


def test_standard_scaler_column_mismatch():
    X_train = np.array([[1, 2], [3, 4]])
    X_test = np.array([[1, 2, 3]])

    scaler = StandardScaler()
    scaler.fit(X_train)

    with pytest.raises(ValueError):
        scaler.transform(X_test)


def test_minmax_scaler_invalid_feature_range():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0))


def test_onehot_encoder_column_mismatch():
    X_train = np.array([["Red", "S"], ["Blue", "M"]])
    X_test = np.array([["Red"]])

    encoder = OneHotEncoder()
    encoder.fit(X_train)

    with pytest.raises(ValueError):
        encoder.transform(X_test)


def test_simple_imputer_all_nan_column():
    X = np.array([
        [np.nan, 1],
        [np.nan, 3],
        [np.nan, 5]
    ])

    imputer = Imputer(strategy="mean", fill_value=0)
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out[:, 0], [0, 0, 0])
    assert np.allclose(X_out[:, 1], [1, 3, 5])

