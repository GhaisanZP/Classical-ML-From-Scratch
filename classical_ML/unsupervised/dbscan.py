import numpy as np
from utils.distances import euclidean_distance

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

        Parameters:
        eps : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples : int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit_predict(self, X):
        """
        Perform DBSCAN clustering from features and return cluster labels.
        Labels are 0, 1, 2, ... for valid clusters. -1 represents NOISE.
        """
        n_samples = X.shape[0]
        # Initialize all labels as -1 (Noise/Unclassified)
        self.labels_ = np.full(n_samples, -1)
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0

        for point_idx in range(n_samples):
            # Skip if already visited
            if visited[point_idx]:
                continue

            visited[point_idx] = True

            # Find all neighbors within eps radius
            neighbors = self._region_query(X, point_idx)

            # If not enough neighbors, it remains Noise (-1) for now
            if len(neighbors) < self.min_samples:
                continue

            # Otherwise, a new cluster is born! Expand it.
            self._expand_cluster(X, visited, point_idx, neighbors, cluster_id)
            cluster_id += 1

        return self.labels_

    def _expand_cluster(self, X, visited, point_idx, neighbors, cluster_id):
        """
        Expands the cluster to all densely connected neighbors.
        """
        # Assign the core point to the current cluster
        self.labels_[point_idx] = cluster_id

        i = 0
        # We use a while loop because 'neighbors' list might grow dynamically
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._region_query(X, neighbor_idx)

                # If the neighbor is also a core point, add its neighbors to our queue
                if len(new_neighbors) >= self.min_samples:
                    for n in new_neighbors:
                        if n not in neighbors:
                            neighbors.append(n)

            # If the neighbor isn't part of any cluster yer (could be previously marked as noise)
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id

            i += 1

    def _region_query(self, X, point_idx):
        """
        Find all data points within the 'eps' radius of a given point.
        """
        neighbors = []
        for idx in range(X.shape[0]):
            if euclidean_distance(X[point_idx], X[idx]) <= self.eps:
                neighbors.append(idx)
        return neighbors