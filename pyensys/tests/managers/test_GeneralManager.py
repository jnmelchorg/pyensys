from os.path import join, dirname
import time

from pyensys.managers.GeneralManager import main_access_function, save_in_json
from pyensys.tests.test_data_paths import get_attest_json_test_data, \
    set_pandapower_test_output_directory


def test_main_access_function():
    start = time.time()
    # main_access_function(file_path=get_attest_json_test_data())
    path = dirname(__file__)
    path = join(path, "..", "json", "attest_input_format_m1_samelinestest1.json")
    solution = main_access_function(file_path=path)
    path = dirname(__file__)
    path = join(path, "..", "outputs")
    output_path = join(path, "output.json")
    save_in_json(solution, output_path)
    end = time.time()
    print(end - start)

test_main_access_function()
