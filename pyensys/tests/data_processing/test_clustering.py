from numpy.core.records import array
from pandas import read_excel, DataFrame
from os.path import dirname, abspath, join
from sklearn.cluster import Birch
from numpy import ndarray,array_equal, allclose, array
from typing import List

from pyensys.data_processing.clustering import Birch_Settings, Clustering, \
    TimeSeriesClustering, PropertiesofScenarioElement

from pyensys.tests.tests_data_paths import get_clustering_test_data

def read_normalized_demand_profiles() -> dict:
    path = get_clustering_test_data()
    return read_excel(io=path, sheet_name=['Sheet1'])

def get_row(data: DataFrame, row: int) -> DataFrame:
    return data.iloc[row]

def process_row(row: DataFrame) -> ndarray:
    row_as_numpy_array = row.to_numpy()
    return row_as_numpy_array.reshape(-1, 1)

def get_training_data() -> ndarray:
    data = read_normalized_demand_profiles()
    row = get_row(data['Sheet1'], 45)
    return process_row(row)

def initialise_birch_clustering_algorithm() -> Clustering:
    clusters = Clustering()
    birch_algorithm_settings = Birch_Settings()
    clusters.set_up_birch_algorithm(birch_algorithm_settings)
    return clusters

def set_time_series(clustering: TimeSeriesClustering) -> TimeSeriesClustering:
    data = read_normalized_demand_profiles()
    clustering.set_time_series(data['Sheet1'])
    return clustering




def test_initialise_birch_clustering_algorithm():
    clusters = Clustering()
    birch_algorithm_settings = Birch_Settings()
    clusters.set_up_birch_algorithm(birch_algorithm_settings)
    assert isinstance(clusters.algorithm, Birch)

def test_train_birch_algorithm():
    clusters = initialise_birch_clustering_algorithm()
    data = get_training_data()
    clusters.train_birch_algorithm(data)
    assert clusters.algorithm.fit_ == True

def test_perform_birch_clustering():
    clusters = initialise_birch_clustering_algorithm()
    data = get_training_data()
    clusters.train_birch_algorithm(data)
    assert array_equal(clusters.perform_clustering(data), [0, 1, 2, 2, 3, 4, 3, 5])

def test_set_time_series():
    data = read_normalized_demand_profiles()
    time_clustering = TimeSeriesClustering()
    time_clustering.set_time_series(data['Sheet1'])
    assert isinstance(time_clustering.time_series, DataFrame)

def test_row_dataframe_to_numpy():
    TEST_ROW: int = 45
    time_clustering = TimeSeriesClustering()
    time_clustering = set_time_series(time_clustering)
    row = time_clustering.time_series.iloc[TEST_ROW]
    row = time_clustering._process_row(row)
    assert isinstance(row, ndarray)

def test_initialise_birch_clustering_algorithm():
    time_clustering = TimeSeriesClustering()
    time_clustering.initialise_birch_clustering_algorithm(Birch_Settings())
    assert time_clustering.clusters.algorithm.threshold == 0.03

def test_perform_time_series_clustering():
    time_clustering = TimeSeriesClustering()
    time_clustering = set_time_series(time_clustering)
    time_clustering.initialise_birch_clustering_algorithm(Birch_Settings())
    time_clustering.perform_clustering()
    assert len(time_clustering.time_series_clusters) != 0

def test_calculate_clusters_centroids():
    time_clustering = TimeSeriesClustering()
    time_clustering = set_time_series(time_clustering)
    time_clustering.initialise_birch_clustering_algorithm(Birch_Settings())
    time_clustering.perform_clustering()
    time_clustering.calculate_clusters_centroids()
    assert len(time_clustering.clusters_centroids) != 0

def test_single_run_birch_algorithm():
    time_clustering = TimeSeriesClustering()
    time_clustering = set_time_series(time_clustering)
    time_clustering.initialise_birch_clustering_algorithm(Birch_Settings())
    TEST_ROW: int = 45
    row = time_clustering.time_series.iloc[TEST_ROW]
    assert array_equal(time_clustering._run_birch_algorithm(row), 
        [0, 1, 2, 2, 3, 4, 3, 5])

def test_enumerate_array():
    TEST_ARRAY: ndarray = array([1, 1, 1])
    time_clustering = TimeSeriesClustering()
    assert time_clustering._enumerate_array(TEST_ARRAY) == [[1, 0], [1, 1], [1, 2]]

def test_sort_array_by_specific_column():
    TEST_ARRAY: List[List[int]] = [[2, 0], [0, 1], [1, 2]]
    time_clustering = TimeSeriesClustering()
    assert time_clustering._sort_array_by_specific_column(TEST_ARRAY, column=0) == \
        [[0, 1], [1, 2], [2, 0]]

def test_add_scenario_to_clusters_list_in_same_cluster():
    TEST_GROUPED_CLUSTERS_VALUES: List[List[float]] = [[0.5, 0.8]]
    TEST_SCENARIO_PROPERTIES: PropertiesofScenarioElement = \
        PropertiesofScenarioElement(cluster_number=0, scenario_number= 2, value= 0.7)
    TEST_CURRENT_CLUSTER: int = 0
    time_clustering = TimeSeriesClustering()
    groups_values, current_scenario = time_clustering._add_scenario_to_clusters_list(\
        TEST_SCENARIO_PROPERTIES, TEST_CURRENT_CLUSTER, TEST_GROUPED_CLUSTERS_VALUES)
    assert groups_values == [[0.5, 0.8, 0.7]]
    assert current_scenario == 0

def test_add_scenario_to_clusters_list_in_new_cluster():
    TEST_GROUPED_CLUSTERS_VALUES: List[List[float]] = [[0.5, 0.8]]
    TEST_SCENARIO_PROPERTIES: PropertiesofScenarioElement = \
        PropertiesofScenarioElement(cluster_number=1, scenario_number= 2, value= 0.7)
    TEST_CURRENT_CLUSTER: int = 0
    time_clustering = TimeSeriesClustering()
    groups_values, current_scenario = time_clustering._add_scenario_to_clusters_list(\
        TEST_SCENARIO_PROPERTIES, TEST_CURRENT_CLUSTER, TEST_GROUPED_CLUSTERS_VALUES)
    assert groups_values == [[0.5, 0.8], [0.7]]
    assert current_scenario == 1

def test_get_values_by_cluster():
    TEST_ROW: int = 45
    time_clustering = TimeSeriesClustering()
    time_clustering = set_time_series(time_clustering)
    TEST_ROW_DATA = time_clustering.time_series.iloc[TEST_ROW]
    TEST_SORTED_CLUSTERS: List[List[int]] = [[0, 0], [1, 1], [2, 2], [2, 3], [3, 5], \
        [4, 4], [4, 6], [5, 7]]
    RESULT: List[List[float]] = time_clustering._get_values_by_cluster(TEST_SORTED_CLUSTERS, \
        TEST_ROW_DATA)
    EXPECTED_RESULTS: List[List[float]] = [[2.278206034], [2.066304312], \
        [1.724012709, 1.780746071], [1.544900112], [1.288527329, 1.256387823], \
        [1.082996991]]
    for real, expected in zip(RESULT, EXPECTED_RESULTS):
        assert allclose(real, expected)

def test_calculate_average_per_cluster():
    time_clustering = TimeSeriesClustering()
    VALUES_BY_CLUSTER: List[List[float]] = [[2.278206034], [2.066304312], \
        [1.724012709, 1.780746071], [1.544900112], [1.288527329, 1.256387823], \
        [1.082996991]]
    EXPECTED_RESULTS = [2.278206034, 2.066304312, 1.75237939, 1.544900112, \
        1.272457576, 1.082996991]
    RESULT: List[float] = time_clustering._calculate_average_per_cluster( \
        VALUES_BY_CLUSTER)
    assert allclose(RESULT, EXPECTED_RESULTS)

def test_assign_centroids():
    time_clustering = TimeSeriesClustering()
    PAIR_CLUSTER_SCENARIO_LIST: List[List[int]] = [[0, 0], [0, 1], [1, 2]]
    TEST_AVERAGE_VALUES_PER_CLUSTER: List[float] = [0.8, 0.5]
    EXPECTED_RESULTS: List[float] = [0.8, 0.8, 0.5]
    RESULTS = time_clustering._assign_centroids(PAIR_CLUSTER_SCENARIO_LIST, \
        TEST_AVERAGE_VALUES_PER_CLUSTER)
    assert allclose(RESULTS, EXPECTED_RESULTS)

def test_calculate_single_time_step_centroids():
    TEST_ROW: int = 45
    time_clustering = TimeSeriesClustering()
    time_clustering = set_time_series(time_clustering)
    TEST_ROW_DATA = time_clustering.time_series.iloc[TEST_ROW]
    TEST_CLUSTER_IDS: ndarray = array([0, 1, 2, 2, 4, 3, 4, 5])
    EXPECTED_RESULTS: List[float] = [2.278206034, 2.066304312, 1.75237939,\
        1.75237939, 1.272457576, 1.544900112, 1.272457576, 1.082996991]
    assert allclose(time_clustering._calculate_single_time_step_centroids(\
        TEST_CLUSTER_IDS, TEST_ROW_DATA), EXPECTED_RESULTS)
    
    
    
    
