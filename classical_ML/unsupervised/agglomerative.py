import numpy as np
from utils.distances import euclidean_distance

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        """
        Agglomerative Hierarchical Clustering (Bottom-Up approach).

        Parameters:
        n_clusters : int
            The number of clusters to find
        linkage : str
            Which linkage criterion to use.
            Options: 'single' (minimum distance), 'complete' (maximum distance), 'average'.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit_predict(self, X):
        """
        Fits the hierarchical clustering and returns the cluster labels.
        """
        n_samples = X.shape[0]

        # 1. Initialize: Every single data point is its own cluster at the beginning
        # We store clusters as a list of lists containing the indices of data points
        clusters = [[i] for i in range(n_samples)]

        # 2. Iterate until we merge enough clusters to reach the target 'n_clusters'
        while len(clusters) > self.n_clusters:
            min_dist = float('inf')
            merge_idx1, merge_idx2 = -1, -1

            # Find the two closest clusters to merge
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._calculate_cluster_distance(X, clusters[i], clusters[j])

                    if dist < min_dist:
                        min_dist = dist
                        merge_idx1 = i
                        merge_idx2 = j

            # 3. Merge cluster j into cluster i
            clusters[merge_idx1].extend(clusters[merge_idx2])

            # 4. Remove cluster j from out list of active clusters
            clusters.pop(merge_idx2)

        # 5. Assign the final cluster labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels_[point_idx] = cluster_id

        return self.labels_

    def _calculate_cluster_distance(self, X, cluster1, cluster2):
        """
        Calculates the distance between two clusters based on the chosen linkage method.
        """
        # Calculate all pairwise distances between points in cluster1 and cluster 2
        distances = [euclidean_distance(X[i], X[j]) for i in cluster1 for j in cluster2]

        if self.linkage == 'single':
            return min(distances)
        elif self.linkage == 'complete':
            return max(distances)
        elif self.linkage == 'average':
            return sum(distances) / len(distances)
        else:
            raise ValueError("Unsupported linkage method. Choose 'single', 'complete', or 'average'")