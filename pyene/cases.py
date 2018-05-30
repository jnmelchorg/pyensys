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
    print('\n\nOF: ', NModel.OF.expr())
    NM.print(NModel)


# Interaction node
class _node():
    def __init__(self):
        self.value = None
        self.index = None
        self.bus = None


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

    # Several RES nodes
    resInNode = _node()
    for xr in range(conf.NoRES):
        resInNode.value = RESProfs[xr][:]
        resInNode.index = xr+1
        EN.loadRES(resInNode)

    # Several hydro nodes
    hydroInNode = _node()
    for xh in range(conf.NoHydro):
        hydroInNode.value = 1000
        hydroInNode.index = xh+1
        EN.loadHydro(hydroInNode)

    # Run integrated pyene
    mod = EN.run()

    # Print results
#    EN.NM.offPrint()
#    EN.NM.Print['Generation'] = True
    EN.Print_ENSim(mod, EN.EM, EN.NM)

    # Collect unused hydro:
    print()
    hydroOutNode = _node()
    for xh in range(EN.EM.size['Vectors']):
        hydroOutNode.index = xh+1
        hydroOutNode.value = EN.getHydro(mod, hydroOutNode)
        print('Hydro %d left: %f' % (hydroOutNode.index, hydroOutNode.value))

    # Collect output of pumps
    print()
    pumpNode = _node()
    for xp in range(EN.NM.pumps['Number']):
        pumpNode.index = xp+1
        pumpNode.value = EN.getPump(mod, xp+1)
        print('Pump %d: %f' % (pumpNode.index, pumpNode.value))

    # Collect RES spill
    print()
    resOutNode = _node()
    for xp in range(EN.NM.RES['Number']):
        resOutNode.index = xp+1
        resOutNode.value = EN.getRES(mod, resOutNode)
        print('RES %d: %f' % (resOutNode.index, resOutNode.value))

    # Collect curtailment
    print()
    curNode = _node()
    for xn in range(EN.NM.networkE.number_of_nodes()):
        curNode.bus = xn+1
        curNode.value = EN.getCurt(mod, curNode)
        print('RES %d: %f' % (curNode.bus, curNode.value))
