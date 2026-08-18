import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        Principal Component Analysis (PCA) for Dimensionality Reduction.
        
        Parameters:
        n_components : int
            Number of principal components to keep.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X):
        """
        Computes the principal components of the dataset.
        """
        # 1. Mean centering (Shift data so that the mean of each feature is 0)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Calculate the Covariance Matrix
        # np.cov expects features as rows, so we set rowvar=False to treat columns as features
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Calculate Eigenvalues and Eigenvectors
        # eigh is optimized for symmetric matrices like covariance matrices
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvectors by decreasing eigenvalues
        # np.linalg.eigh returns ascending order, so we reverse it [::-1]
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 5. Store the first n_components eigenvectors (These are our Principal Components)
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

    def transform(self, X):
        """
        Applies dimensionality reduction to X by projecting it onto the principal components.
        """
        # Mean center the data using the training mean
        X_centered = X - self.mean
        
        # Project the data using the dot product
        return np.dot(X_centered, self.components)
        
    def fit_transform(self, X):
        """Fits the model and then transforms the data."""
        self.fit(X)
        return self.transform(X)