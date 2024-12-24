import numpy as np
from scipy.stats import mode

class KPrototypesCustom:
    def __init__(self, n_clusters=4, max_iter=100, gamma=1.0):
        """
        :param n_clusters: Number of clusters.
        :param max_iter: Maximum number of iterations.
        :param gamma: Weight for the categorical distance.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.gamma = gamma
        self.centroids = None

    def fit(self, X_num, X_cat):
        """
        Train the model with numeric and categorical data.
        """
        n_samples = X_num.shape[0]

        #----- Initialize centroids randomly
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids_num = X_num[indices]
        centroids_cat = X_cat[indices]

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X_num, X_cat, centroids_num, centroids_cat)
            new_centroids_num, new_centroids_cat = self._update_centroids(X_num, X_cat, labels)

            if np.all(centroids_num == new_centroids_num) and np.all(centroids_cat == new_centroids_cat):
                break

            centroids_num, centroids_cat = new_centroids_num, new_centroids_cat

        self.centroids = (centroids_num, centroids_cat)
        self.labels_ = labels

    def _assign_clusters(self, X_num, X_cat, centroids_num, centroids_cat):
        """
        Assign data points to the closest cluster.
        """
        distances = np.zeros((X_num.shape[0], self.n_clusters))
        for i, (c_num, c_cat) in enumerate(zip(centroids_num, centroids_cat)):
            dist_num = np.linalg.norm(X_num - c_num, axis=1)
            dist_cat = np.sum(X_cat != c_cat, axis=1)
            distances[:, i] = dist_num + self.gamma * dist_cat
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X_num, X_cat, labels):
        """
        Calculate new centroids based on the assignments.
        """
        centroids_num = np.zeros((self.n_clusters, X_num.shape[1]))
        centroids_cat = np.zeros((self.n_clusters, X_cat.shape[1]), dtype=X_cat.dtype)

        for cluster in range(self.n_clusters):
            mask = labels == cluster
            if np.any(mask):
                centroids_num[cluster] = X_num[mask].mean(axis=0)
                centroids_cat[cluster] = mode(X_cat[mask], axis=0).mode[0]
        return centroids_num, centroids_cat

    def predict(self, X_num, X_cat):
        """
        Predict the cluster of new data points.
        """
        if self.centroids is None:
            raise ValueError("The model is not trained.")
        centroids_num, centroids_cat = self.centroids
        distances = np.zeros((X_num.shape[0], self.n_clusters))
        for i, (c_num, c_cat) in enumerate(zip(centroids_num, centroids_cat)):
            dist_num = np.linalg.norm(X_num - c_num, axis=1)
            dist_cat = np.sum(X_cat != c_cat, axis=1)
            distances[:, i] = dist_num + self.gamma * dist_cat
        return np.argmin(distances, axis=1)
