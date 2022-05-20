from os.path import join

from pyensys.managers.GeneralManager import main_access_function, save_in_json
from pyensys.tests.test_data_paths import get_attest_json_test_data, set_pandapower_test_output_directory


def test_main_access_function():
    # main_access_function(file_path=get_attest_json_test_data())
    solution = main_access_function(
        file_path="C:\\Users\\f09903jm\\git projects\\pyensys\\pyensys\\tests\\json\\attest_input_format_m1.json")
    output_dir = "C:\\Users\\f09903jm\\git projects\\pyensys\\pyensys\\tests\\outputs"
    output_path = join(output_dir, "output.json")
    save_in_json(solution, output_path)

test_main_access_function()
