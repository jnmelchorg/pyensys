from pyensys.managers.GeneralManager import main_access_function
from pyensys.tests.tests_data_paths import get_path_pandapower_json_test_data

def test_main_access_function():
    main_access_function(file_path=get_path_pandapower_json_test_data())
