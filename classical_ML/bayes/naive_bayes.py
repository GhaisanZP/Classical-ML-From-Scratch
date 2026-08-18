import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        """
        Gaussian Naive Bayes Classifier.
        Assumes continuous features follow a Gaussian (Normal) distribution.
        """
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Calculate the mean, variance, and prior probability for each class.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize arrays
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            # Isolate data for the current class
            X_c = X[y == c]

            # Calculate mean and variance per feature for the class
            self.mean[idx, :] = X_c.mean(axis=0)

            # Add a tiny epsilon (1e-9) to variance to prevent division by zero later
            self.var[idx, :] = X_c.var(axis=0) + 1e-9

            # Prior probability P(y) = (count of class / total count)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def _pdf(self, class_idx, x):
        """
        Gaussian Probability Density Function (PDF).
        Formula: (1 / sqrt(2 * pi * var)) * exp(- (x - mean)^2 / (2 * var))
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]

        numerator = np.exp(-0.5 * ((x - mean) ** 2) / var)
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_single(self, x):
        """
        Predicts the class for a single instance using log probabilities.
        """
        posteriors = []
        for idx, c in enumerate(self.classes):
            # Calculate log prior: log(P(y))
            prior = np.log(self.priors[idx])

            # Calculate log likelilhood: \sum log(P(x|y))
            # We use log to prevent numerical underflow when multiplying many small probabilities
            posterior = np.sum(np.log(self._pdf(idx, x)))

            # Bayes theorem numerator (in log space, multiplication becomes addition)
            posterior = prior + posterior
            posteriors.append(posterior)

        # Return the class with the highest posterior probability (Maximum A Posteriori)
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        """
        Predicts class labels for given input data X.
        """
        predicted_labels = [self._predict_single(x) for x in X]
        return np.array(predicted_labels)
