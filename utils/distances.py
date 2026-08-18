import numpy as np

def euclidean_distance(x1, x2):
    """
    Computes the Euclidean distance between two vectors.
    Formula: \sqrt{\sum_{i=1}^{n} (x1_i - x2_i)^2}
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    """
    Computes the Manhattan distance between two vectors.
    Formula: \sum_{i=1}^{n} |x1_i - x2_i|
    """
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p):
    """
    Computes the Minkowski distance between two vectors.
    Formula: (\sum_{i=1}^{n} |x1_i - x2_i|^p)^{1/p}
    """
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)