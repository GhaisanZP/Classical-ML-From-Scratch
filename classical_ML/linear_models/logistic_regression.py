import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, threshold=0.5):
        """
        Logistic Regression Model optimized via Gradient Descent / Ascent.

        Parameters:
        learning_rate : float
            Step size for the Gradient Descent.
        n_iters : int
            Number of iterations for optimization.
        threshold : float
            Probability threshold for binary classification.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        Applies the Sigmoid activation function.
        Formula: \sigma(z) = \frac{1}{1 + e^{-z}}
        """
        # Clip z to prevent overflow in np.exp
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Trains the Logistic Regression model using Maximum Likelihood / Gradient Descent.
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Optimization loop
        for _ in range(self.n_iters):
            # Linear model: z = X * w + b
            linear_model = np.dot(X, self.weights) + self.bias

            # Apply Sigmoid: p = 1 / (1 + exp^{-z})
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients based on Log-Loss (Cross-Entropy)
            # \partial J / \partial w = (1/N) * X^T * (y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))

            # \partial J / \partial b = (1/N) * \sum (y_pred - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Predicts the probability of the positive class (class 1).
        Formula: P(y=1|X) = \sigma(X * w + b)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        """
        Predicts the class labels (0 or 1) based on the defined threshold.
        """
        y_predicted_cls = [1 if i > self.threshold else 0 for i in self.predict_proba(X)]
        return np.array(y_predicted_cls)