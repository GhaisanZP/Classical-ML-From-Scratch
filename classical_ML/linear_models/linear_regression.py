import numpy as np

class LinearRegression:
    def __init__(self, method='lse', learning_rate=0.01, n_iters=1000):
        """
        Linear Regression Model.

        Parameters:
        method : str
            'lse' for Least Squared Estimation (Exact mathematical solution).
            'gd' for Gradient Descent (Iterative optimization).
        learning_rate : float
            Step size for Gradient Descent.
        n_iters : int
            Number of iterations for Gradient Descent.
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Trains the model using the selected method.
        """
        n_samples, n_features = X.shape

        if self.method == 'lse':
            # Add bias term (column of 1s) to X
            # X_b matrix dimension: (n_samples, n_features + 1)
            X_b = np.c[np.ones((n_samples, 1)), X]

            # LSE Formula: \theta = (X^T * X)^{-1} * X^T * y
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

            self.bias = theta[0]
            self.weights = theta[1:]

        elif self.method == 'gd':
            # Initiative parameters
            self.weights = np.zeros(n_features)
            self.bias = 0

            # Gradient Descent iteration
            for _ in range(self.n_iters):
                y_predicted = np.dot(X, self.weights) + self.bias

                # Compute gradients
                # \partial J / \partial w = (1/N) * X^T * (y_pred - y)
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
                # \partial J / \partial b = (1/N) * \sum (y_pred - y)
                db = (1 / n_samples) * np.sum(y_predicted - y)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        else:
            raise ValueError("Invalid method. Choose 'lse' or 'gd'")

    def predict(self, X):
        """
        Predicts target values for given input data.
        Formula: \hat{y} = X * w + b
        """
        return np.dot(X, self.weights) + self.bias