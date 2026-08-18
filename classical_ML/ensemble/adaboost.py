import numpy as np
from classical_ml.tree_based.decision_tree import DecisionTree

class AdaBoost:
    def __init__(self, n_estimators=10):
        """
        AdaBoost (Adaptive Boosting) Classifier.
        
        Parameters:
        n_estimators : int
            Number of weak learners (decision stumps) to train.
        """
        self.n_estimators = n_estimators
        self.clfs = []
        self.alphas = []

    def fit(self, X, y):
        """
        Trains the AdaBoost ensemble sequentially.
        """
        n_samples, n_features = X.shape
        
        # Initialize weights to 1/N (All samples are equally important at the start)
        w = np.full(n_samples, (1 / n_samples))
        
        self.clfs = []
        self.alphas = []
        
        for _ in range(self.n_estimators):
            # We use Decision Stumps (Decision Tree with max_depth=1) as weak learners
            clf = DecisionTree(max_depth=1)
            
            # Sample data according to sample weights (Han et al. 2022 approach)
            # This allows us to use our unweighted DecisionTree implementation
            indices = np.random.choice(n_samples, n_samples, p=w, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            clf.fit(X_sample, y_sample)
            
            # Predict on the original training set to calculate weighted error
            preds = clf.predict(X)
            
            # Calculate error: sum of weights of misclassified samples
            misclassified = (preds != y)
            error = np.sum(w[misclassified])
            
            # If error is >= 0.5, the weak learner is worse than random guessing. Abort loop.
            if error >= 0.5 or error == 0:
                break
                
            # Calculate alpha (the weight/importance of this specific classifier)
            # Formula: 0.5 * ln((1 - error) / error)
            alpha = 0.5 * np.log((1.0 - error) / error)
            
            # Update weights
            # AdaBoost formula expects labels and predictions to be -1 or 1
            y_ = np.where(y == 0, -1, 1)
            preds_ = np.where(preds == 0, -1, 1)
            
            # Increase weights for misclassified, decrease for correctly classified
            w *= np.exp(-alpha * y_ * preds_)
            
            # Normalize weights so they sum to 1
            w /= np.sum(w)
            
            self.clfs.append(clf)
            self.alphas.append(alpha)

    def predict(self, X):
        """
        Predicts class for X by taking a weighted sum of predictions from all weak learners.
        """
        # Get predictions from each weak learner
        clf_preds = np.array([clf.predict(X) for clf in self.clfs])
        
        # Map predictions from 0/1 to -1/1 for the weighted sum
        clf_preds_ = np.where(clf_preds == 0, -1, 1)
        
        # Calculate the weighted sum of predictions: \sum \alpha_t * h_t(x)
        y_pred = np.dot(self.alphas, clf_preds_)
        
        # Convert back to 0/1 labels (sign function)
        return np.where(y_pred < 0, 0, 1)