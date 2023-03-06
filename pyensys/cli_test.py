# from pyensys.cli import pyensys_entry_point
# from pyensys.tests.test_data_paths import get_path_pandapower_json_test_data
# from click.testing import CliRunner

# print('get_path_pandapower_json_test_data:')
# print(get_path_pandapower_json_test_data)

# def test_pyensys_entry_point():
#     runner = CliRunner()
#     runner.invoke(pyensys_entry_point,
#                   [get_path_pandapower_json_test_data()])
    
from pyensys.cli import cli
from click.testing import CliRunner


# result = CliRunner.invoke(cli, ['--debug', 'sync'])
# result = CliRunner.invoke(cli, ['run-dist_invest'])

runner = CliRunner()
# result = runner.invoke(cli, input = 'run-dist_invest')
result = runner.invoke(cli, ['run-dist_invest'])
assert result.exit_code == 0

print(result)
# cli("--hydro")