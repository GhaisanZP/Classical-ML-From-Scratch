import numpy as np
from utils.distances import euclidean_distance

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        """
        K-Means Clustering Algorithm.

        Parameters:
        k : int
            The number of clusters to form as well as the number of centroids to generate.
        max_iters : int
            Maximum number of iteration of the k-means algorithm for a signle run.
        tol : float
            Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = []
        self.clusters = []

    def fit(self, X):
        """
        Compute k-means clustering.
        """
        n_samples, n_features = X.shape

        # 1. Initialize centroids randomly by picking k existing data points
        random_sample_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_sample_idxs]

        for _ in range(self.max_iters):
            # 2. Assign sammples to the closest centroids to create clusters
            self.clusters = self._create_clusters(X)

            # Save current centroids for convergence check
            centroids_old = self.centroids.copy()

            # 3. Calculate new centroids from the clusters
            self._update_centroids(X)

            # 4. Check for convergence (if centroids have not moved significantly)
            if self._is_converged(centroids_old, self.centroids):
                break

    def _create_clusters(self, X):
        """
        Assigns each sample to the closest centroid.
        """
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(X):
            centroid_idx = self._closest_centroid(sample)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample):
        """
        Calculates distance from a sample to each centroid and returns the closest index.
        """
        distances = [euclidean_distance(sample, point) for point in self.centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _update_centroids(self, X):
        """
        Updates centroids as the mean of the samples in each cluster.
        """
        for cluster_idx, cluster in enumerate(self.clusters):
            # Handle empty clusters to prevent division by zero
            if len(cluster) == 0:
                continue

            cluster_mean = np.mean(X[cluster], axis=0)
            self.centroids[cluster_idx] = cluster_mean

    def _is_converged(self, centroids_old, centroids_new):
        """
        Checks if the distance between old and new centroids is less than tolerance.
        """
        distances = [euclidean_distance(centroids_old[i], centroids_new[i]) for i in range(self.k)]
        return sum(distances) < self.tol

    def predict(self, X):
        """
        Predicts the closest cluster each sample in X belongs to.
        """
        labels = np.zeros(X.shape[0], dtype=int)
        for idx, sample in enumerate(X):
            centroid_idx = self._closest_centroid(sample)
            labels[idx] = centroid_idx
        return labels