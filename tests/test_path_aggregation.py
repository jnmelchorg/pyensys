from pandas import read_excel, DataFrame
from os.path import dirname, abspath, join
from sklearn.cluster import Birch
from numpy import ndarray,array_equal

from pyensys.data_processing.path_aggregation import Birch_Settings, Clustering

def get_test_file_path() -> str:
    current_directory_path=dirname(abspath(__file__))
    excel_file_path = 'excel\\normalized_demand_profiles.xlsx'
    return join(current_directory_path, excel_file_path)

def read_normalized_demand_profiles() -> dict:
    path = get_test_file_path()
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
