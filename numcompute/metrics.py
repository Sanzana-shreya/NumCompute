import numpy as np

# For Classification Metrics 
# Accuracy
def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

# Confusion Matrix
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)

    cm = np.zeros((n, n), dtype=int)

    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            cm[i, j] = np.sum((y_true == c1) & (y_pred == c2))

    return cm

# Precision
def precision(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return tp / (tp + fp + 1e-9)

# Recall
def recall(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp / (tp + fn + 1e-9)

# F1 Score
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-9)


# Regression Metrics

def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)




# Root Mean Squared Error (RMSE)
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Mean Absolute Error (MAD / MAE)
def mad(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


# Mean Absolute Percentage Error (MAPE)
def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # avoid division by zero
    non_zero = y_true != 0
    y_true = y_true[non_zero]
    y_pred = y_pred[non_zero]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# For ROC Curve (Binary) 

def roc_curve(y_true, y_scores):
    
    #y_scores: probability scores (not labels)
    
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # sort by scores descending
    desc = np.argsort(-y_scores)
    y_true = y_true[desc]

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    tp_total = tp[-1]
    fp_total = fp[-1]

    tpr = tp / (tp_total + 1e-9)
    fpr = fp / (fp_total + 1e-9)

    return fpr, tpr


# For AUC 

def auc(fpr, tpr):

    # Trapezoidal rule
    
    return np.trapezoid(tpr, fpr)