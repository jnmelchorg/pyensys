""" Test the pyeneE engine. """
from fixtures import testConfig, json_directory
from pyene.engines.pyene import pyeneClass as pe
import os


# Small network
def test_pyeneN_4Bus():
    print('test_pyeneN_4Bus: case4.json')
    conf = testConfig()
    conf.NM.settings['File'] = os.path.join(json_directory(), 'case4.json')
    EN = pe(conf.EN)
    # Initialise model
    (NM, NModel, results) = EN.NSim(conf)
    NM.print(NModel)
    print(NModel.OF.expr())

    assert 0.0001 >= abs(NModel.OF.expr()-21952.5)


# Medium network
def test_pyeneN_14Bus():
    print('test_pyeneN_14Bus: case14.json')
    conf = testConfig()
    conf.NM.settings['File'] = os.path.join(json_directory(), 'case14.json')
    EN = pe(conf.EN)
    # Initialise model
    (NM, NModel, results) = EN.NSim(conf)
    NM.print(NModel)

    assert 0.0001 >= abs(NModel.OF.expr()-7704.910967)
