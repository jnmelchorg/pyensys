""" Test the pyeneE engine. """
from pyene.fixtures import testConfig, json_directory
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


# Small network - Security and losses
def test_pyeneN_4BusSec():
    print('test_pyeneN_4BusSec: case4.json')
    conf = testConfig()
    conf.NM.settings['File'] = os.path.join(json_directory(), 'case4.json')
    conf.NM.settings['Security'] = [2, 3]
    conf.NM.settings['Losses'] = True
    conf.NM.settings['NoTime'] = 2
    conf.NM.scenarios['Demand'] = [1, 1.1]
    conf.NM.settings['Feasibility'] = True

    EN = pe(conf.EN)

    # Initialise model
    (NM, NModel, results) = EN.NSim(conf)
    NM.print(NModel)
    print('Losses')
    Lss01 = NModel.vNLoss[0, 0].value*NM.ENetwork.data['baseMVA']
    Lss02 = NModel.vNLoss[3, 1].value*NM.ENetwork.data['baseMVA']
    print(Lss01, Lss02)
    print(NModel.OF.expr())

    assert (0.0001 >= abs(NModel.OF.expr()-63215.0917360614) and
            0.0001 >= abs(Lss01 - 0.10718868) and
            0.0001 >= abs(Lss02 - 1.56890072)
            )

