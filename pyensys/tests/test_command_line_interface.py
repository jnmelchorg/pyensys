from pyensys.cli import pyensys_entry_point
from pyensys.tests.test_data_paths import get_path_pandapower_json_test_data
from click.testing import CliRunner


def test_pyensys_entry_point():
    runner = CliRunner()
    runner.invoke(pyensys_entry_point,
                  [get_path_pandapower_json_test_data()])
