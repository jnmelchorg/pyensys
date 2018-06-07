""" This module contains the FutureDAMS - pyene test case definitions.

These test cases are used to assess the different functionalities of the
engine.

# TODO complete this description once the cases are written.

"""
from .engines.pyene import pyeneClass as pe
import numpy as np
import os

# Energy balance test
def test_pyeneE(config):
    """ Execute pyene to access pyeneE - Full json based simulation."""
    EN = pe()
    (EM, EModel, results) = EN.ESim(config)
    EM.print(EModel)


# Network simulation test
def test_pyeneN(config):
    """ Execute pyene to access pyeneN - Full json based simulation."""
    # Create object
    EN = pe()
    # Run model
    (NM, NModel, results) = EN.NSim(config)
    print('\n\nOF: ', NModel.OF.expr())
    NM.print(NModel)


# Interaction node
class _node():
    def __init__(self):
        self.value = None
        self.index = None
        self.bus = None
        self.marginal = None
        self.flag = False


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
        hydroInNode.value = 0
        hydroInNode.index = xh+1
        EN.loadHydro(hydroInNode)

    # Run integrated pyene
    mod = EN.run()

    # Print results
    print('\n\nOF: ', mod.OF.expr())
    EN.NM.offPrint()
    EN.NM.Print['Generation'] = True
    EN.NM.Print['Losses'] = True
    EN.Print_ENSim(mod, EN.EM, EN.NM)

    # Collect unused hydro:
    print()
    hydroOutNode = _node()
    for xh in range(EN.EM.size['Vectors']):
        hydroOutNode.index = xh+1
        hydroOutNode = EN.getHydro(mod, hydroOutNode)
        print('Hydro %d left: %f (%f)' % (hydroOutNode.index,
              hydroOutNode.value, hydroOutNode.marginal), hydroOutNode.flag)

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

    # Collect curtailment per node
    print()
    curNode = _node()
    for xn in range(EN.NM.networkE.number_of_nodes()):
        curNode.bus = xn+1
        curNode.value = EN.getCurt(mod, curNode)
        print('Dem %d: %f' % (curNode.bus, curNode.value))

    # Collect all curtailment
    print()
    curAll = _node()
    curAll.value = EN.getCurtAll(mod)
    print('Total curtailment:', curAll.value)


# pyene test
def test_pyenetest():
    '''Test specific functionalities'''
    # Initialize configuration
    conf = default_conf()
    # Selected network file
    conf.NetworkFile = 'case14.json'
    # Location of the json directory
    conf.json = conf.json = os.path.join(os.path.dirname(__file__), 'json')
    # Define number of time spets
    conf.Time = 5  # Number of time steps
    # Hydropower
    conf.NoHydro = 2  # Number of hydropower plants
    conf.Hydro = [1, 2]  # Location (bus) of hydro
    conf.HydroMax = [100, 100]  # Generation capacity
    conf.HydroCost = [0.01, 0.01]  # Costs

    # Pumps
    conf.NoPump = 1  # Number of pumps
    conf.Pump = [3]  # Location (bus) of pumps
    conf.PumpMax = [1000]  # Generation capacity
    conf.PumpVal = [0.001]  # Value/Profit

    # RES generators
    conf.NoRES = 3  # Number of RES generators
    conf.RES = [3, 4, 5]  # Location (bus) of pumps
    conf.RESMax = [500, 500, 500]  # Generation capacity
    conf.Cost = [0.0001, 0.0001, 0.0001]  # Costs

    # Get Pyene model
    EN = pe()
    # Initialize network model using the selected configuration
    EN.initialise(conf)
    # Convert to pypsa
    xscen = 0  # Selected scenario
    EN.pyene2pypsa(xscen)
    
# pyene simulation test
def get_pyene():
    """ Get pyene object."""

    return pe()


class default_conf():
    def __init__(self):
        self.init = False  # skip file reading?
        self.TreeFile = 'ResolutionTreeMonth01.json'  # Selected tree file
        self.NetworkFile = 'case4.json'  # Selected network file
        self.json = None  # Location of the json directory

        # Hydropower
        self.NoHydro = 0  # Number of hydropower plants
        self.Hydro = []  # Location (bus) of hydro
        self.HydroMax = []  # Generation capacity
        self.HydroCost = []  # Costs

        # Pumps
        self.NoPump = 0  # Number of pumps
        self.Pump = []  # Location (bus) of pumps
        self.PumpMax = []  # Generation capacity
        self.PumpVal = []  # Value/Profit

        # RES generators
        self.NoRES = 0  # Number of RES generators
        self.RES = []  # Location (bus) of pumps
        self.RESMax = []  # Generation capacity
        self.Cost = []  # Costs

        # Network considerations
        self.Security = []  # List of contingescies to test
        self.Losses = False  # Model losses
        self.Feasibility = False  # Add dummy generators
        self.Time = 0  # Number of time steps

        # Scenarios
        self.NoDemProfiles = 2  # Number of demand profiles
        self.NoRESProfiles = 2  # Number of RES profiles
