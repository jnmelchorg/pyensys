from pyensys.readers.ReaderManager import read_parameters
from pyensys.tests.tests_data_paths import get_path_pandapower_json_test_data


def test_read_parameters_case_json():
    file_path = get_path_pandapower_json_test_data()
    parameters = read_parameters(file_path)
    assert parameters.initialised

def test_read_parameters_case_unknown_extension():
    file_path = 'test.unknown'
    parameters = read_parameters(file_path)
    assert not parameters.initialised