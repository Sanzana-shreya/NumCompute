import numpy as np
import pytest

from numcompute.preprocessing import (
    Imputer,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    ColumnTransformer,
)


# ------------------------------------------------------------
# StandardScaler tests
# ------------------------------------------------------------

def test_standard_scaler_basic():
    """
    Test that StandardScaler standardises each column to:
    - mean approximately 0
    - standard deviation approximately 1
    """
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Assert column-wise mean is close to 0
    assert np.allclose(np.mean(X_scaled, axis=0), [0, 0])

    # Assert column-wise standard deviation is close to 1
    assert np.allclose(np.std(X_scaled, axis=0), [1, 1])


def test_standard_scaler_constant_column():
    """
    Test that a constant column is transformed safely.
    Since variance is zero, scaled output should be all zeros.
    """
    X = np.array([[5], [5], [5]], dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled, [[0], [0], [0]])


def test_standard_scaler_transform_before_fit():
    """
    Error-handling test:
    transform() should raise ValueError if called before fit().
    """
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.transform(np.array([[1, 2]], dtype=float))


def test_standard_scaler_partial_fit():
    """
    Test streaming/incremental update using partial_fit().
    Only shape is checked here to confirm transform still works after multiple chunks.
    """
    X1 = np.array([[1, 2], [3, 4]], dtype=float)
    X2 = np.array([[5, 6]], dtype=float)

    scaler = StandardScaler()
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    X_scaled = scaler.transform(X1)

    assert X_scaled.shape == X1.shape


# ------------------------------------------------------------
# MinMaxScaler tests
# ------------------------------------------------------------

def test_minmax_scaler_basic():
    """
    Test that MinMaxScaler maps values into [0, 1].
    """
    X = np.array([[1], [3], [5]], dtype=float)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled, [[0], [0.5], [1]])


def test_minmax_scaler_constant_column():
    """
    Test that a constant column is handled safely.
    Since max == min, output should become all zeros.
    """
    X = np.array([[5], [5], [5]], dtype=float)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled, [[0], [0], [0]])


def test_minmax_scaler_partial_fit():
    """
    Test streaming/incremental update using partial_fit().
    """
    X1 = np.array([[1], [3]], dtype=float)
    X2 = np.array([[5]], dtype=float)

    scaler = MinMaxScaler()
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    X_scaled = scaler.transform(X1)

    assert X_scaled.shape == X1.shape


# ------------------------------------------------------------
# Imputer tests
# ------------------------------------------------------------

def test_simple_imputer_mean():
    """
    Test mean imputation on a simple numeric column.
    Missing value should be replaced with the column mean.
    """
    X = np.array([[1.0], [np.nan], [3.0]])
    imputer = Imputer(strategy="mean")
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out, [[1.0], [2.0], [3.0]])


def test_simple_imputer_median():
    """
    Test median imputation on a simple numeric column.
    Missing value should be replaced with the column median.
    """
    X = np.array([[1.0], [np.nan], [5.0]])
    imputer = Imputer(strategy="median")
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out, [[1.0], [3.0], [5.0]])


def test_simple_imputer_constant():
    """
    Test constant-value imputation.
    Missing values should be replaced by the user-provided fill_value.
    """
    X = np.array([[1.0], [np.nan], [5.0]])
    imputer = Imputer(strategy="constant", fill_value=-1)
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out, [[1.0], [-1.0], [5.0]])


def test_simple_imputer_partial_fit():
    """
    Test incremental imputation with partial_fit().

    Two chunks are provided:
    - X1 contains one valid value: 1.0
    - X2 contains one valid value: 3.0

    Running mean after both chunks should be:
        (1.0 + 3.0) / 2 = 2.0

    Therefore, transforming [[np.nan]] should produce [[2.0]].
    """
    X1 = np.array([[1.0], [np.nan]])
    X2 = np.array([[3.0], [np.nan]])

    imputer = Imputer(strategy="mean")
    imputer.partial_fit(X1)
    imputer.partial_fit(X2)

    X_out = imputer.transform(np.array([[np.nan]]))

    assert np.allclose(X_out, [[2.0]])
    
def test_simple_imputer_invalid_strategy():
    """
    Error-handling test:
    unsupported imputation strategies should raise ValueError.
    """
    with pytest.raises(ValueError):
        Imputer(strategy="mode")


def test_simple_imputer_all_nan_column():
    """
    Test that a column containing only NaN values is handled safely.
    Such a column should fall back to fill_value.
    """
    X = np.array([
        [np.nan, 1],
        [np.nan, 3],
        [np.nan, 5]
    ], dtype=float)

    imputer = Imputer(strategy="mean", fill_value=0)
    X_out = imputer.fit_transform(X)

    # First column is all NaN, so it should be replaced by fill_value=0
    assert np.allclose(X_out[:, 0], [0, 0, 0])

    # Second column has no missing values and should remain unchanged
    assert np.allclose(X_out[:, 1], [1, 3, 5])


# ------------------------------------------------------------
# OneHotEncoder tests
# ------------------------------------------------------------

def test_onehot_encoder_basic():
    """
    Test one-hot encoding of a simple 2-column categorical dataset.
    Output should contain only 0/1 values.
    """
    X = np.array([
        ["Red", "S"],
        ["Blue", "M"],
        ["Red", "M"],
        ["Green", "S"]
    ], dtype=object)

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    assert X_encoded.shape == (4, 5)
    assert np.all((X_encoded == 0) | (X_encoded == 1))


def test_onehot_encoder_transform_before_fit():
    """
    Error-handling test:
    transform() should raise ValueError if called before fit().
    """
    encoder = OneHotEncoder()

    with pytest.raises(ValueError):
        encoder.transform(np.array([["Red"]], dtype=object))


def test_onehot_encoder_partial_fit():
    """
    Test streaming/incremental category learning with partial_fit().
    """
    X1 = np.array([["Red"], ["Blue"]], dtype=object)
    X2 = np.array([["Green"]], dtype=object)

    encoder = OneHotEncoder()
    encoder.partial_fit(X1)
    encoder.partial_fit(X2)

    X_encoded = encoder.transform(X1)

    assert X_encoded.shape[0] == X1.shape[0]


def test_onehot_encoder_column_mismatch():
    """
    Error-handling test:
    transforming data with a different number of columns should raise ValueError.
    """
    X_train = np.array([["Red", "S"], ["Blue", "M"]], dtype=object)
    X_test = np.array([["Red"]], dtype=object)

    encoder = OneHotEncoder()
    encoder.fit(X_train)

    with pytest.raises(ValueError):
        encoder.transform(X_test)


# ------------------------------------------------------------
# ColumnTransformer tests
# ------------------------------------------------------------

def test_column_transformer_mixed_data():
    """
    Test ColumnTransformer on mixed numeric + categorical data.
    Numeric column -> impute + scale
    Categorical column -> one-hot encode
    """
    X = np.array([
        [25, "Male"],
        [30, "Female"],
        [22, "Female"]
    ], dtype=object)

    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])
    X_out = ct.fit_transform(X)

    assert X_out.shape == (3, 3)


def test_column_transformer_with_nan_numeric():
    """
    Test ColumnTransformer with missing values in numeric columns.
    Output should remain finite after imputation and scaling.
    """
    X = np.array([
        [25, "Male"],
        [np.nan, "Female"],
        [35, "Female"]
    ], dtype=object)

    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])
    X_out = ct.fit_transform(X)

    assert np.all(np.isfinite(X_out.astype(float)))


def test_column_transformer_partial_fit():
    """
    Test streaming/incremental update for ColumnTransformer.
    """
    X1 = np.array([
        [25, "Male"],
        [30, "Female"]
    ], dtype=object)

    X2 = np.array([
        [22, "Female"]
    ], dtype=object)

    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])
    ct.partial_fit(X1)
    ct.partial_fit(X2)

    X_out = ct.transform(X1)

    assert X_out.shape[0] == X1.shape[0]


def test_column_transformer_no_columns_selected():
    """
    Error-handling test:
    if neither numeric nor categorical columns are selected,
    fit_transform() should raise ValueError.
    """
    X = np.array([[1, "A"], [2, "B"]], dtype=object)
    ct = ColumnTransformer(num_cols=[], cat_cols=[])

    with pytest.raises(ValueError):
        ct.fit_transform(X)


# ------------------------------------------------------------
# General error-handling / validation tests
# ------------------------------------------------------------

def test_wrong_dimension_input():
    """
    Error-handling test:
    StandardScaler expects 2D input. A 1D array should raise ValueError.
    """
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.fit(np.array([1, 2, 3], dtype=float))


def test_standard_scaler_column_mismatch():
    """
    Error-handling test:
    transforming data with a different number of columns should raise ValueError.
    """
    X_train = np.array([[1, 2], [3, 4]], dtype=float)
    X_test = np.array([[1, 2, 3]], dtype=float)

    scaler = StandardScaler()
    scaler.fit(X_train)

    with pytest.raises(ValueError):
        scaler.transform(X_test)


def test_minmax_scaler_invalid_feature_range():
    """
    Error-handling test:
    invalid feature_range where min >= max should raise ValueError.
    """
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0))