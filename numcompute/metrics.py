import numpy as np


# Validation Helper
def _validate_inputs(y_true, y_pred=None):
    """
    Validate inputs for metrics.

    - Ensures arrays are numeric
    - Ensures same shape for y_true and y_pred
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
            raise ValueError("y_true and y_pred must have the same shape")

    return y_true, y_pred


# For Classification Metrics 

# Accuracy
def accuracy(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(y_true == y_pred)


# Confusion Matrix
def confusion_matrix(y_true, y_pred):
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
def precision(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    # Assumes binary classification (0 and 1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        raise ValueError("No positive predictions; precision undefined")

    return tp / (tp + fp)


# Recall
def recall(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    # Assumes binary classification (0 and 1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        raise ValueError("No actual positives; recall undefined")

    return tp / (tp + fn)


# F1 Score
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        raise ValueError("Precision and Recall are zero; F1 undefined")

    return 2 * p * r / (p + r)


# Regression Metrics

def mse(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)


# Root Mean Squared Error (RMSE)
def rmse(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Mean Absolute Error (MAD / MAE)
def mad(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


# Mean Absolute Percentage Error (MAPE)
def mape(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    # avoid division by zero
    non_zero = y_true != 0

    if not np.any(non_zero):
        raise ValueError("All y_true values are zero; MAPE undefined")

    y_true = y_true[non_zero]
    y_pred = y_pred[non_zero]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# For ROC Curve (Binary) 

def roc_curve(y_true, y_scores):
    
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

def auc(fpr, tpr):

    # Trapezoidal rule
    
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")

    if len(fpr) < 2:
        raise ValueError("At least two points required to compute AUC")

    return np.trapezoid(tpr, fpr)