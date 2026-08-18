import numpy as np

class RegressionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeRegressor:
    """
    A Decision Tree for Regression.
    Uses Variance Reduction (MSE) instead of Entropy to find the best splits.
    """
    def __init__(self, min_samples_split=2, max_depth=3):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        # Stopping criteria: Max depth reached, too few samples, or pure node
        if (depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1):
            return RegressionNode(value=np.mean(y)) # Left value is the mean of residuals

        best_feat, best_thresh = self._best_split(X, y, n_features)

        # If no valid split is found
        if best_feat is None:
            return RegressionNode(value=np.mean(y))

        left_idxs, right_idxs = self._split(X[:,best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return RegressionNode(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, n_features):
        best_var_red = -1
        split_idx, split_thresh = None, None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                var_red = self._variance_reduction(y, X_column, thr)
                if var_red > best_var_red:
                    best_var_red = var_red
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _variance_reduction(self, y, X_column, threshold):
        """
        Calculate Variance Reduction (equivalent to reducing MSE).
        """
        parent_var = np.var(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        child_var = (len(left_idxs)/n) * np.var(y[left_idxs]) + (len(right_idxs)/n) * np.var(y[right_idxs])

        # We want to maximize the reduction in variance
        return parent_var - child_var

    def _split(self, X_column, split_thresh):
        return np.argwhere(X_column <= split_thresh).flatten(), np.argwhere(X_column > split_thresh).flatten()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class GradientBoosting:
    def __init__(self, n_estimator=10, learning_rate=0.1, max_depth=3):
        """
        Gradient Boosting Classifier.
        Iteratively trains Regression Trees on the pseudo-residuals of the previous models.
        """
        self.n_estimator = n_estimator
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.F0 = 0 # Initial log-odds prediction

    def _sigmoid(self, x):
        # Clip x to prevent overflow
        x = np.clip(x, -250, 250)
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        # Initial predition (F0) is the log-odds of the positive class
        p = np.clip(np.mean(y), 1e-9, 1 - 1e-9)
        self.F0 = np.log(p / (1 - p))

        # Fm holds the cumulative predictions (in log-odds space)
        Fm = np.full(len(y), self.F0)

        for _ in range(self.n_estimator):
            # 1. Calculate current predicted probabilities
            preds = self._sigmoid(Fm)

            # 2. Calculate pseudo_residuals (Gradient of Log-odds)
            residuals = y - preds

            # 3 Fit a Regression Tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # 4. Update the cumulative predictions
            # Note: A full Friedman GBM calculates optimal gamma at leaves, 
            # but a simplified gradient step uses the tree output directly multiplied by learning rate.
            update = tree.predict(X)
            Fm += self.learning_rate * update

    def predict_proba(self, X):
        """
        Predicts probability of class 1.
        """
        Fm = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            Fm += self.learning_rate * tree.predict(X)
        return self._sigmoid(Fm)

    def predict(self, X):
        """
        Thresholds probabilities at 0.5 to return binary labels."""
        probs = self.predict_proba(X)
        return np.where(probs >= 0.5, 1, 0)