from pyensys.cli import pyensys_entry_point
from click.testing import CliRunner

def test_pyensys_entry_point():
    runner = CliRunner()
    result = runner.invoke(pyensys_entry_point, 
        ['C:\\Users\\f09903jm\\git projects\\pyensys\\tests\\json\\pandapower_test_conf.json'])
    assert result.exit_code == 0