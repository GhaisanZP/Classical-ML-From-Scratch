import numpy as np
from collections import Counter
from classical_ml.tree_based.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        """
        Random Forest Classifier using Bagging (Bootstrap Aggregation).

        Parameters:
        n_trees : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of each decision tree.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        n_features : int
            The number of features to consider when looking for the best split.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """
        Builds a forest of trees from the training set (X, y) using Bootstrap samples.
        """
        self.trees = []
        for _ in range(self.n_trees):
            # Initialize a new Decision Tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            # Create a bootstrap sample
            X_sample, y_sample = self._bootstrap_samples(X, y)

            # Train the tree on the bootstrap sample
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        """
        Generates a random sample from the dataset with replacement (Bootstrap).
        """
        n_samples = X.shape[0]
        # np.random.choice with replace=True is the core of Bagging
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _majority_vote(self, y):
        """
        Returns the most common class label.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        Predicts class for X by taking the majority vote from all trees in the forest.
        """
        # Get predictions from each tree
        # tree_preds shape: (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # Swap axes to shape: (n_samples, n_trees) to easily iterate over each sample's predictions
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        # Take the majority vote for each sample
        y_pred = [self._majority_vote(sample_preds) for sample_preds in tree_preds]
        return np.array(y_pred)