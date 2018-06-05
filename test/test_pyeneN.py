""" Test the pyeneE engine. """
from click.testing import CliRunner
from fixtures import *
from pyene.engines.pyene import pyeneClass as pe


# Small network
def test_pyeneN_4Bus():
    config = get_config()
    print('test_pyeneN_4Bus: case4.json')
    config.NetworkFile = 'case4.json'
    EN = pe()
    # Initialise model
    (NM, NModel, results) = EN.NSim(config)    
    NM.print(NModel)

    assert NModel.OF.expr() == 21952.5

# Medium network
def test_pyeneN_14Bus():
    config = get_config()
    print('test_pyeneN_14Bus: case14.json')
    config.NetworkFile = 'case14.json'
    EN = pe()
    # Initialise model
    (NM, NModel, results) = EN.NSim(config)    
    NM.print(NModel)

    print('%.30f ' % NModel.OF.expr())
    assert 0.0001 >= abs(NModel.OF.expr()-7704.910967)
