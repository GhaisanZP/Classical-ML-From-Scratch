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