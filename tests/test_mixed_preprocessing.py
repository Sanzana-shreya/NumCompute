# testing separate preprocessing for numeric and categorical data
import numpy as np
import pytest

from numcompute.preprocessing import StandardScaler, OneHotEncoder


def test_separate_preprocessing_and_combination():
    # Numeric data
    X_num = np.array([
        [25],
        [30],
        [22]
    ], dtype=float)

    # Categorical data
    X_cat = np.array([
        ["Male"],
        ["Female"],
        ["Female"]
    ], dtype=object)

    scaler = StandardScaler()
    encoder = OneHotEncoder()

    # Apply transformations
    X_num_scaled = scaler.fit_transform(X_num)
    X_cat_encoded = encoder.fit_transform(X_cat)

    # combining scaled numeric and encoded categorical data
    X_final = np.hstack((X_num_scaled, X_cat_encoded))

    # Check shapes match number of samples
    assert X_num_scaled.shape[0] == X_cat_encoded.shape[0]
    assert X_final.shape[0] == X_num.shape[0]

    # Check numeric scaling (mean ~0)
    assert np.allclose(np.mean(X_num_scaled, axis=0), 0)

    # Check encoded output is numeric
    assert np.issubdtype(X_cat_encoded.dtype, np.number)

    # Final output should also be numeric
    assert np.issubdtype(X_final.dtype, np.number)


def test_shape_mismatch_error():
    # Numeric data with 3 samples
    X_num = np.array([
        [25],
        [30],
        [22]
    ], dtype=float)

    # Categorical data with mismatched samples (2 instead of 3)
    X_cat = np.array([
        ["Male"],
        ["Female"]
    ], dtype=object)

    scaler = StandardScaler()
    encoder = OneHotEncoder()

    X_num_scaled = scaler.fit_transform(X_num)
    X_cat_encoded = encoder.fit_transform(X_cat)

    # combining scaled numeric and encoded categorical data
    # This should raise an error due to mismatched row sizes
    with pytest.raises(ValueError):
        np.hstack((X_num_scaled, X_cat_encoded))


def test_standard_scaler_invalid_input():
    # Invalid scaler input
    X_num = np.array([25, 30, 22], dtype=float)

    scaler = StandardScaler()

    # Should raise error because input is not 2D
    with pytest.raises(ValueError):
        scaler.fit_transform(X_num)


def test_onehot_encoder_invalid_input():
    # Invalid encoder input
    X_cat = np.array(["Male", "Female"], dtype=object)

    encoder = OneHotEncoder()

    # Should raise error because input is not 2D
    with pytest.raises(ValueError):
        encoder.fit_transform(X_cat)