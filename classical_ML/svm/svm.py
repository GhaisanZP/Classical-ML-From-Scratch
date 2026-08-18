import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Support Vector Machine (SVM) Classifier using Gradient Descent (Hinge Loss).

        Parameters:
        learning_rate: float
            Step size for gradient descent.
        lambda_param: float
            Regularization parameter (equivalent to 1/C).
            Controls the trade-off between maximizing the margin and minimizing the classification error.
        n_iters: int
            Number of iterations for optimization.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None  
        self.bias = None  

    def fit(self, X, y):
        """
        Train the SVM model using Stochastic Gradient Descent.
        """
        n_samples, n_features = X.shape

        # Ensure labels are -1 and 1 for SVM mathematical formulation
        y_ = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent using Hinge Loss
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check if the margin condition is satisfied: y_i * (w.x_i - b) >= 1
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1

                if condition:
                    # If strictly inside the margin or correctly classified, only regularize weights
                    # \partial J / \partial w = 2 * lambda * w    
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # If margin is violated, update weights and bias to correct the error
                    # \partial J / \partial w = 2 * lambda * w - y_i * x_i
                    # \partial J / \partial b = -y_i
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(y_[idx], x_i))
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        """
        Predict class labels (0 or 1).
        Formula: sign(X * w + b)
        """
        linear_output = np.dot(X, self.weights) - self.bias
        # The sign function returns -1 or 1, we convert -1 back to 0 for standard binary classification
        y_pred = np.sign(linear_output)
        y_pred = np.where(y_pred == -1, 0, 1)
        return y_pred