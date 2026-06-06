# For metrics.py

import numpy as np
import pytest

from numcompute_stream.metrics import (
    accuracy, confusion_matrix,
    precision, recall, f1,
    mse, rmse, mad, mape,
    roc_curve, auc
)


def test_classification_metrics():
    print("\n===== METRICS TESTS =====")

    # Normal Classification
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    # Check accuracy
    assert np.isclose(accuracy(y_true, y_pred), 0.8)

    # Check confusion matrix shape
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)

    # Precision, Recall, F1 should be valid floats
    assert 0 <= precision(y_true, y_pred) <= 1
    assert 0 <= recall(y_true, y_pred) <= 1
    assert 0 <= f1(y_true, y_pred) <= 1


def test_precision_no_positive_predictions():
    # Edge Case: No positive predictions
    with pytest.raises(ValueError):
        precision([1, 1, 1], [0, 0, 0])


def test_recall_no_actual_positives():
    # Edge Case: No actual positives
    with pytest.raises(ValueError):
        recall([0, 0, 0], [0, 0, 0])


def test_shape_mismatch_accuracy():
    # Shape mismatch
    with pytest.raises(ValueError):
        accuracy([1, 0], [1, 0, 1])


def test_empty_accuracy():
    # Empty input
    with pytest.raises(ValueError):
        accuracy([], [])


def test_non_numeric_accuracy():
    # Non-numeric input
    with pytest.raises(ValueError):
        accuracy(["a", "b"], ["a", "b"])


def test_regression_metrics():
    # Regression Tests
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])

    assert mse(y_true, y_pred) >= 0
    assert rmse(y_true, y_pred) >= 0
    assert mad(y_true, y_pred) >= 0
    assert mape(y_true, y_pred) >= 0


def test_mape_zero_case():
    # MAPE Zero Division Case
    with pytest.raises(ValueError):
        mape([0, 0, 0], [1, 2, 3])


def test_roc_curve_valid():
    # ROC Curve
    y_true = np.array([0, 1, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

    fpr, tpr = roc_curve(y_true, y_scores)

    # Check valid outputs
    assert len(fpr) == len(tpr)
    assert np.all(fpr >= 0)
    assert np.all(tpr >= 0)


def test_roc_invalid_labels():
    # ROC Invalid Case (non-binary)
    with pytest.raises(ValueError):
        roc_curve([0, 1, 2], [0.1, 0.5, 0.9])


def test_roc_empty_input():
    # ROC empty input
    with pytest.raises(ValueError):
        roc_curve([], [])


def test_roc_shape_mismatch():
    # ROC shape mismatch
    with pytest.raises(ValueError):
        roc_curve([0, 1], [0.2])


def test_auc_shape_mismatch():
    # AUC Shape Mismatch
    with pytest.raises(ValueError):
        auc([0, 0.5], [0.1])


def test_auc_empty_input():
    # AUC empty input
    with pytest.raises(ValueError):
        auc([], []) 