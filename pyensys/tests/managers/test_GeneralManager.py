from pyensys.managers.GeneralManager import main_access_function
from pyensys.tests.test_data_paths import get_attest_json_test_data


def test_main_access_function():
    main_access_function(file_path=get_attest_json_test_data())
