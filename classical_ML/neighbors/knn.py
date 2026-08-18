import numpy as np
from collections import Counter
from utils.distances import euclidean_distance, manhattan_distance, minkowski_distance

class KNN:
    def __init__(self, k=5, metric='euclidean', p=2):
        """
        K-Nearest Neighbors Classifier (Lazy Learner).

        Parameters:
        k : int
            Number of nearest neighbors to use for majority voting.
        metric : str
            Distance metric to use ('euclidean', 'manhattan', 'minkowski').
        p : int
            The power parameter for the Minkowski distance.
        """
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fits the model. For k-NN, this simply stores the training data.
        """
        self.X_train = X
        self.y_train = y

    def _calculate_distance(self, x1, x2):
        """
        Helper method to route to the correct distance function.
        """
        if self.metric == 'euclidean':
            return euclidean_distance(x1, x2)
        elif self.metric == 'manhattan':
            return manhattan_distance(x1, x2)
        elif self.metric == 'minkowski':
            return minkowski_distance(x1, x2, self.p)
        else:
            raise ValueError("Unsupported distance metric.")

    def _predict_single(self, x):
        """
        Predicts the class for a single instance.
        """
        # 1. Calculate distances from x and all examples in the training set
        distances = [self._calculate_distance(x, x_train) for x_train in self.X_train]

        # 2. Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]

        # 3. Extract the labels of the k nearest neighbors training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 4. Return the most common class label (majority voting)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """
        Predicts class labels for given input data X.
        """
        predicted_labels =np.array([self._predict_single(x) for x in X])
        return np.array(predicted_labels)