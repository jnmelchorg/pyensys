from sklearn.cluster import Birch
from dataclasses import dataclass
from numpy import ndarray


@dataclass
class Birch_Settings:
    threshold : float = 0.03
    branching_factor : int = 50
    n_clusters: int = 0

class Clustering:

    def _check_number_clusters(self, settings: Birch_Settings):
        if settings.n_clusters > 0:
            return settings.n_clusters
        else:
            return None

    def set_up_birch_algorithm(self, settings: Birch_Settings):
        number_clusters = self._check_number_clusters(settings)
        self.algorithm = Birch(threshold = settings.threshold, 
            branching_factor= settings.branching_factor,
            n_clusters=number_clusters)

    def train_birch_algorithm(self, data: ndarray):
        self.algorithm.fit(data)

    def perform_clustering(self, data: ndarray):
        return self.algorithm.fit_predict(data)

