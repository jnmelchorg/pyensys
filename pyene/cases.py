""" This module contains the FutureDAMS - pyene test case definitions.

These test cases are used to assess the different functionalities of the
engine.

# TODO complete this description once the cases are written.

"""
from .engines.pyene import pyeneClass as pe
import numpy as np

# Energy balance test
def test_pyeneE(config):
    """ Execute pyene to access pyeneE - Full json based simulation."""
    EN = pe()
    # Avoid loading file
    if config.init:
        EN.fRea = False
    (EM, EModel, results) = EN.ESim(config.TreeFile)
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

# Interaction node
class _node(object):
    _properties = {
            'value': None,
            'index': None,
            'bus': None,
            'scenario': None
             }

# pyene simulation test
def test_pyene(conf):
    """ Execute pyene to access pyeneN - Full json based simulation."""
    # Create object
    EN = pe()

    # Initialise with selected configuration
    EN.initialise(conf)

    # Fake weather engine
    FileName = 'TimeSeries.json'
    (DemandProfiles, NoDemPeriod, BusDem, LinkDem, NoRES, NoRESP,
     LLRESType, LLRESPeriod, RESProfs, RESBus, RESLink, NoLink,
     Nohr) = EN.ReadTimeS(FileName)

    # Single demand node (first scenario)
    demandNode = _node()
    demandNode.value = DemandProfiles[0][:]
    demandNode.index = 1
    EN.loadDemand(demandNode)

    # Second scenario
    demandNode = _node()
    demandNode.value = DemandProfiles[1][:]
    demandNode.index = 2
    EN.loadDemand(demandNode)

#    # Several RES nodes
#    resNode = _node()
#    for xr in range(conf.NoRES):
#        resNode.value = RESProfs[xr][:]
#        resNode.index = xr+1
#        EN.loadRES(resNode)

#    # Several hydro nodes
#    hydroNode = _node()
#    for xh in range(conf.NoHydro):
#        hydroNode.value = 100
#        hydroNode.index = xh+1
#        EN.loadHydro(hydroNode)

    # Run integrated pyene
    mod = EN.run()
#    print('0  -  Gen: ', mod.vGen[0, 0].value)
#    for x in range(10):
#        print(x+1,x,'Gen:', mod.vGen[x+1, 0].value, ' Cst:', mod.vGCost[x, 0].value)
#    print('11  -  Gen: ', mod.vGen[11, 0].value)
#    for x in range(12, 20):
#        print(x,x-2,'Gen:', mod.vGen[x,0].value, ' Cst:', mod.vGCost[x-2, 0].value)
#
#    aux[1000]
    # Print results
    EN.Print_ENSim(mod, EN.EM, EN.NM)

    # Collect output of pumps
    indexPump=1
    pumpNode = _node()
    pumpNode.value = EN.getPump(mod, indexPump)
    pumpNode.index =indexPump

    
