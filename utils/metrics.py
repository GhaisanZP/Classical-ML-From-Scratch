import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE).
    Formula: (1/n) * \sum (y_true - y_pred)^2
    """
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE).
    Formula: (1/n) * \sum |y_true - y_pred|
    """
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """
    Calculates the R^2 (Coefficient of Determination).
    Formula: 1 - (SS_res / SS_tot)
    where SS_res is the residual sum of squares and
    SS_tot is the total sum of squares.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)

def accuracy_score(y_true, y_pred):
    """
    Calculates the Accuracy classification score for binary classification.
    Formula: TP / (TP + TN + FP + FN)
    where TP is True Positives, TN is True Negatives, FP is False Positives, and FN is False Negatives.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + tn + fp + fn == 0:
        return 0.0

    return (tp + tn) / (tp + tn + fp + fn)

def precision_score(y_true, y_pred):
    """
    Calculates the Precision score for binary classification.
    Formula: TP / (TP + FP)
    where TP is True Positives and FP is False Positives.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    """
    Calculates the Recall score for binary classification.
    Formula: TP / (TP + FN)
    where TP is True Positives and FN is False Negatives.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    """
    Calculates the F1 score for binary classification.
    Formula: 2 * (Precision * Recall) / (Precision + Recall)
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)