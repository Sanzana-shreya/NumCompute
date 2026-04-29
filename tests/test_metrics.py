# For metrics.py

import numpy as np
from numcompute.metrics import (
    accuracy, confusion_matrix,
    precision, recall, f1,
    mse, rmse, mad, mape,
    roc_curve, auc
)

print("\n===== METRICS TESTS =====")

# Normal Classification 

y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0])

print("Accuracy:", accuracy(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Precision:", precision(y_true, y_pred))
print("Recall:", recall(y_true, y_pred))
print("F1:", f1(y_true, y_pred))


# Edge Case: No positive predictions

try:
    precision([1, 1, 1], [0, 0, 0])
except ValueError as e:
    print("\nPrecision Error:", e)


# Edge Case: No actual positives

try:
    recall([0, 0, 0], [0, 0, 0])
except ValueError as e:
    print("Recall Error:", e)


# Shape mismatch 

try:
    accuracy([1, 0], [1, 0, 1])
except ValueError as e:
    print("Shape Error:", e)


# Regression Tests

y_true = np.array([10, 20, 30])
y_pred = np.array([12, 18, 33])

print("\nRegression:")
print("MSE:", mse(y_true, y_pred))
print("RMSE:", rmse(y_true, y_pred))
print("MAD:", mad(y_true, y_pred))
print("MAPE:", mape(y_true, y_pred))


# MAPE Zero Division Case

try:
    mape([0, 0, 0], [1, 2, 3])
except ValueError as e:
    print("MAPE Error:", e)


# ROC Curve

y_true = np.array([0, 1, 1, 0, 1])
y_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

fpr, tpr = roc_curve(y_true, y_scores)
print("\nROC:")
print("FPR:", fpr)
print("TPR:", tpr)


# AUC 

print("AUC:", auc(fpr, tpr))


# ROC Invalid Case (non-binary)

try:
    roc_curve([0, 1, 2], [0.1, 0.5, 0.9])
except ValueError as e:
    print("ROC Error:", e)


# AUC Shape Mismatch

try:
    auc([0, 0.5], [0.1])
except ValueError as e:
    print("AUC Error:", e)