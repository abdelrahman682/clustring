import numpy as np
import matplotlib.pyplot as plt
import os 
os.system("cls")


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__(self, n_clusters = 3, max_iters = 100, tol = 1e4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        self.tol = tol

    def fit(self, x):
        self.x = x
        self.n_sampled, self.n_features = x.shape
        
        #initial value for centriods
        random_idxs = np.random.choice(self.n_sampled, self.n_clusters, replace=False)
        self.centroids = [self.x[idx] for idx in random_idxs]
        
        for _ in range(self.max_iters):
            # Assign samples to nearest centriod
            self.labels = self._create_clusters(self.centroids)

            # update centriods
            old_centroids = self.centroids
            self.centroids = self._update_centroids(self.labels)

            # check covergwend
            if self._is_converged(old_centroids, self.centroids):
                break


    def _create_clusters(self, centriods):
        clusters = [[] for _ in range(self.n_clusters)]
        # Assign samples to nearest centriod
        for idx, value in enumerate(self.x):
            centroid_idx = self._nearest_centroid(value, centriods)
            clusters[centroid_idx].append(idx)
        return clusters

    def _nearest_centroid(self, value, centriods):
        distances = [euclidean_distance(value, point) for point in centriods]
        closest_idx = np.argmin(distances)
        return closest_idx 

    def _update_centroids(self, labels):
        centriods = np.zeros((self.n_clusters, self.n_features))
        for idx, cluster in enumerate(labels):
            cluster_mean = np.mean(self.x[cluster], axis=0)
            centriods[idx] = cluster_mean
        return centriods

    def _is_converged(self, old, new):
        distances = [euclidean_distance(old[i], new[i]) for i in range(self.n_clusters)]
        return np.sum(distances) == self.tol
    
    def predict(self, x):
        predicted_label = []
        for i in x:
            idx = self._nearest_centroid(x, self.centroids)
            predicted_label.append(idx)
        return predicted_label