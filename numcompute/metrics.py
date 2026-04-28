import numpy as np
from typing import Tuple


# Validation Helper
def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Validate inputs for metric computations.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,), optional
        Predicted target values.

    Returns
    -------
    y_true : np.ndarray of shape (n_samples,)
        Validated ground truth array.
    y_pred : np.ndarray of shape (n_samples,) or None
        Validated prediction array.

    Raises
    ------
    ValueError
        If inputs are None, non-numeric, or shapes mismatch.

    Notes
    -----
    Ensures inputs are NumPy arrays and numeric.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    if y_true is None:
        raise ValueError("y_true cannot be None")

    y_true = np.asarray(y_true)

    if not np.issubdtype(y_true.dtype, np.number):
        raise ValueError("y_true must be numeric")

    if y_pred is not None:
        y_pred = np.asarray(y_pred)

        if not np.issubdtype(y_pred.dtype, np.number):
            raise ValueError("y_pred must be numeric")

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}"
            )

    return y_true, y_pred


# For Classification Metrics 

# Accuracy
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        Fraction of correctly predicted samples.

    Notes
    -----
    Works for binary and multiclass classification.

    Complexity
    ----------
    Time: O(n)
    Space: O(1)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


# Confusion Matrix
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    np.ndarray of shape (n_classes, n_classes)
        Confusion matrix where rows correspond to true labels
        and columns correspond to predicted labels.

    Notes
    -----
    Supports multiclass classification.

    Complexity
    ----------
    Time: O(n * k^2)
    Space: O(k^2)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))  # all unique labels
    n = len(classes)

    cm = np.zeros((n, n), dtype=int)

    # Rows = true labels, cols = predicted labels
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            cm[i, j] = np.sum((y_true == c1) & (y_pred == c2))

    return cm


# Precision
def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute precision for binary classification.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth binary labels (0 or 1).
    y_pred : np.ndarray of shape (n_samples,)
        Predicted binary labels.

    Returns
    -------
    float
        Precision score.

    Raises
    ------
    ValueError
        If no positive predictions exist.

    Complexity
    ----------
    Time: O(n)
    Space: O(1)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    # Assumes binary classification (0 and 1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        raise ValueError("No positive predictions; precision undefined")

    return float(tp / (tp + fp))


# Recall
def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute recall for binary classification.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth binary labels (0 or 1).
    y_pred : np.ndarray of shape (n_samples,)
        Predicted binary labels.

    Returns
    -------
    float
        Recall score.

    Raises
    ------
    ValueError
        If no actual positives exist.

    Complexity
    ----------
    Time: O(n)
    Space: O(1)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    # Assumes binary classification (0 and 1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        raise ValueError("No actual positives; recall undefined")

    return float(tp / (tp + fn))


# F1 Score
def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        F1 score.

    Raises
    ------
    ValueError
        If precision and recall are both zero.

    Complexity
    ----------
    Time: O(n)
    Space: O(1)
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        raise ValueError("Precision and Recall are zero; F1 undefined")

    return float(2 * p * r / (p + r))


# Regression Metrics

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
    y_pred : np.ndarray of shape (n_samples,)

    Returns
    -------
    float
        Mean squared error.

    Complexity
    ----------
    Time: O(n)
    Space: O(1)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


# Root Mean Squared Error (RMSE)
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Complexity
    ----------
    Time: O(n)
    Space: O(1)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# Mean Absolute Error (MAD / MAE)
def mad(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE).

    Complexity
    ----------
    Time: O(n)
    Space: O(1)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


# Mean Absolute Percentage Error (MAPE)
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Notes
    -----
    Ignores zero values in y_true to avoid division by zero.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    # avoid division by zero
    non_zero = y_true != 0

    if not np.any(non_zero):
        raise ValueError("All y_true values are zero; MAPE undefined")

    y_true = y_true[non_zero]
    y_pred = y_pred[non_zero]

    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


# For ROC Curve (Binary) 

def roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ROC curve for binary classification.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Binary ground truth labels (0 or 1).
    y_scores : np.ndarray of shape (n_samples,)
        Predicted probability scores.

    Returns
    -------
    fpr : np.ndarray
        False positive rates.
    tpr : np.ndarray
        True positive rates.

    Complexity
    ----------
    Time: O(n log n)
    Space: O(n)
    """

    #y_scores: probability scores (not labels)
    
    if y_true is None or y_scores is None:
        raise ValueError("Inputs cannot be None")

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")

    # EnsureS binary labels
    unique = np.unique(y_true)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("y_true must contain only binary labels (0 and 1)")

    # Sort by scores descending
    desc = np.argsort(-y_scores)
    y_true = y_true[desc]

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    tp_total = tp[-1]
    fp_total = fp[-1]

    if tp_total == 0 or fp_total == 0:
        raise ValueError("ROC undefined: need both positive and negative samples")

    tpr = tp / tp_total
    fpr = fp / fp_total

    return fpr, tpr


# For AUC 

def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute Area Under the Curve (AUC) using trapezoidal rule.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates.
    tpr : np.ndarray
        True positive rates.

    Returns
    -------
    float
        AUC score.

    Complexity
    ----------
    Time: O(n)
    Space: O(1)
    """

    # Trapezoidal rule
    
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")

    if len(fpr) < 2:
        raise ValueError("At least two points required to compute AUC")

    return float(np.trapz(tpr, fpr))