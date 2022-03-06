from sklearn.cluster import Birch
from dataclasses import dataclass
from numpy import ndarray, array
from pandas import DataFrame
from statistics import mean

from typing import Tuple, List


@dataclass
class Birch_Settings:
    threshold : float = 0.03
    branching_factor : int = 50
    n_clusters: int = 0

@dataclass
class PropertiesofScenarioElement:
    cluster_number: int
    value: float
    scenario_number: int

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

    def perform_clustering(self, data: ndarray) -> ndarray:
        return self.algorithm.fit_predict(data)

class TimeSeriesClustering:
    def _process_row(self, row: DataFrame) -> ndarray:
        row_as_numpy_array = row.to_numpy()
        return row_as_numpy_array.reshape(-1, 1)
    
    def _run_birch_algorithm(self, row: DataFrame) -> ndarray:
        numpy_row = self._process_row(row)
        self.clusters.train_birch_algorithm(numpy_row)
        return self.clusters.perform_clustering(numpy_row)
    
    def _enumerate_array(self, array: ndarray) -> list:
        enumerated_array = []
        for pos, val in enumerate(array.tolist()):
            enumerated_array.append([val, pos])
        return enumerated_array
    
    def _sort_array_by_specific_column(self, array: list, column: int) -> list:
        return sorted(array, key=lambda x: x[column])
    
    def _add_scenario_to_clusters_list(self, properties: PropertiesofScenarioElement,\
        current_cluster: int, grouped_clusters_values: List[List[float]]) -> \
        Tuple[List[List[float]], int]:
        if properties.cluster_number == current_cluster:
            grouped_clusters_values[current_cluster].append(properties.value)
        else:
            current_cluster = properties.cluster_number
            grouped_clusters_values.append([properties.value])
        return grouped_clusters_values, current_cluster

    def _get_values_by_cluster(self, sorted_clusters: List[List[int]], \
        time_step_data: DataFrame) -> List[List[float]]:
        numpy_time_step_data = self._process_row(time_step_data)
        values_by_cluster = []
        current_cluster = -1
        for element in sorted_clusters:
            properties = PropertiesofScenarioElement(cluster_number=element[0], \
                scenario_number=element[1], value=0.0)
            properties.value = numpy_time_step_data[properties.scenario_number][0]
            values_by_cluster, current_cluster = self._add_scenario_to_clusters_list(\
                properties, current_cluster, values_by_cluster)
        return values_by_cluster
    
    def _calculate_average_per_cluster(self, values_by_cluster: List[float]) -> List[float]:
        average_value_per_cluster = []
        for value in values_by_cluster:
            average_value_per_cluster.append(mean(value))
        return average_value_per_cluster
    
    def _assign_centroids(self, enumerated_clusters: List[List[int]], \
        average_value_per_cluster: List[float]) -> List[float]:
        centroids = [0.0 for _ in enumerated_clusters]
        for element in enumerated_clusters:
            centroids[element[1]] = average_value_per_cluster[element[0]]
        return centroids

    def _calculate_single_time_step_centroids(self, clusters_id_list: ndarray, \
        time_step_data: DataFrame) -> List[float]:
        enumerated_clusters = self._enumerate_array(clusters_id_list)
        sorted_clusters = self._sort_array_by_specific_column(enumerated_clusters, column=0)
        values_by_cluster = self._get_values_by_cluster(sorted_clusters, time_step_data)
        average_value_per_cluster = self._calculate_average_per_cluster(values_by_cluster)        
        return self._assign_centroids(enumerated_clusters, average_value_per_cluster)

    def set_time_series(self, data: DataFrame):
        self.time_series = data
    
    def initialise_birch_clustering_algorithm(self, settings: Birch_Settings):
        self.clusters = Clustering()
        birch_algorithm_settings = settings
        self.clusters.set_up_birch_algorithm(birch_algorithm_settings)
    
    def perform_clustering(self):
        self.time_series_clusters = []
        for _, time_step_data in self.time_series.iterrows():
            self.time_series_clusters.append(self._run_birch_algorithm(time_step_data))
    
    def calculate_clusters_centroids(self):
        self.clusters_centroids: List[List[float]] = []
        for clusters_id_list, (_, time_step_data) in zip(self.time_series_clusters, \
            self.time_series.iterrows()):
            self.clusters_centroids.append(\
                self._calculate_single_time_step_centroids(array(clusters_id_list), \
                time_step_data))
            

            



            





