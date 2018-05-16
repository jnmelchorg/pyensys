""" This module contains the FutureDAMS - pyene test case definitions.

These test cases are used to assess the different functionalities of the
engine.

# TODO complete this description once the cases are written.

"""
from .engines.pyene import pyeneClass as pe


# Energy balance test
def test_pyeneE(config):
    """ Execute pyene to access pyeneE - Full json based simulation."""    
    EN = pe()
    # Avoid loading file
    if config.init:
        EN.fRea = False
    (EM, EModel, results)=EN.ESim(config.TreeFile)
    EM.print(EModel)

# Network simulation test
def test_pyeneN(config):
    """ Execute pyene to access pyeneN - Full json based simulation."""    
    EN = pe()
    # Avoid loading file
    if config.init:
        EN.fRea = False
    (NM, NModel, results) = EN.NSim(config.NetworkFile)
    NM.print(NModel)

# pyene simulation test
def test_pyene(config):
    """ Execute pyene to access pyeneN - Full json based simulation."""    
    EN = pe()
    # Avoid loading file
    if config.init:
        EN.fRea = False

    (mod, EM, NM) = EN.ENSim(config.TreeFile, config.NetworkFile)
    EN.Print_ENSim(mod, EM, NM)
