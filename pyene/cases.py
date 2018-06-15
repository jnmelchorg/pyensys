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
    EN.set_Demand(demandNode.index, demandNode.value)

    # Second scenario
    demandNode = _node()
    demandNode.value = DemandProfiles[1][:]
    demandNode.index = 2
    EN.set_Demand(demandNode.index, demandNode.value)

    # Several RES nodes
    resInNode = _node()
    for xr in range(conf.NoRES):
        resInNode.value = RESProfs[xr][:]
        resInNode.index = xr+1
        EN.set_RES(resInNode.index, resInNode.value)

    # Several hydro nodes
    hydroInNode = _node()
    for xh in range(conf.NoHydro):
        hydroInNode.value = 0
        hydroInNode.index = xh+1
        EN.set_Hydro(hydroInNode.index, hydroInNode.value)

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
        print('Hydro %d', hydroOutNode.index, ' left: ',
              EN.get_Hydro(mod, hydroOutNode.index), ' (',
              EN.get_HydroMarginal(mod, hydroOutNode.index), ')',
              EN.get_HydroFlag(mod, hydroOutNode.index))

    # Collect output of pumps
    print()
    pumpNode = _node()
    for xp in range(EN.NM.pumps['Number']):
        pumpNode.index = xp+1
        pumpNode.value = EN.get_Pump(mod, xp+1)
        print('Pump %d: %f' % (pumpNode.index, pumpNode.value))

    # Collect RES spill
    print()
    resOutNode = _node()
    for xp in range(EN.NM.RES['Number']):
        resOutNode.index = xp+1
        resOutNode.value = EN.get_RES(mod, resOutNode)
        print('RES %d: %f' % (resOutNode.index, resOutNode.value))

    # Collect curtailment per node
    print()
    curNode = _node()
    for xn in range(EN.NM.networkE.number_of_nodes()):
        curNode.bus = xn+1
        curNode.value = EN.get_DemandCurtailment(mod, curNode.bus)
        print('Dem %d: %f' % (curNode.bus, curNode.value))

    # Collect all curtailment
    print()
    curAll = _node()
    curAll.value = EN.get_AllDemandCurtailment(mod)
    print('Total curtailment:', curAll.value)


# pyene test
def test_pyenetest():
    '''Test specific functionalities'''
    # Initialize configuration
    conf = default_conf()
    # Selected network file
#    conf.TreeFile = 'TestCase.json'
    conf.NetworkFile = 'case4.json'
    # Location of the json directory
    conf.json = conf.json = os.path.join(os.path.dirname(__file__), 'json')
    # Consider single time step
    conf.Time = 2  # Number of time steps
    conf.Weights = [0.5, 1]
    # Add hydropower plant
    conf.NoHydro = 1#2  # Number of hydropower plants
    conf.Hydro = [1]#[1, 2]  # Location (bus) of hydro
    conf.HydroMax = [150]#[100, 100]  # Generation capacity
    conf.HydroCost = [0.01]#[0.01, 0.01]  # Costs

    # Pumps
    conf.NoPump = 1  # Number of pumps
    conf.Pump = [2]  # Location (bus) of pumps
    conf.PumpMax = [1000]  # Generation capacity
    conf.PumpVal = [0.001]  # Value/Profit

#    # RES generators
#    conf.NoRES = 3  # Number of RES generators
#    conf.RES = [1, 2, 3]  # Location (bus) of pumps
#    conf.RESMax = [500, 500, 500]  # Generation capacity
#    conf.Cost = [0.0001, 0.0001, 0.0001]  # Costs
    # Enable curtailment
    conf.Feasibility = True
    # Get Pyene model
    EN = pe()
    # Initialize network model using the selected configuration
    EN.initialise(conf)
#    # Fake weather engine
#    FileName = 'TimeSeries.json'
#    (DemandProfiles, NoDemPeriod, BusDem, LinkDem, NoRES, NoRESP,
#     LLRESType, LLRESPeriod, RESProfs, RESBus, RESLink, NoLink,
#     Nohr) = EN.ReadTimeS(FileName)

    # Single demand node (first scenario)
    demandNode = _node()
    demandNode.value = [0.2, 0.1]  # DemandProfiles[0][0:conf.Time]
    demandNode.index = 1
    EN.set_Demand(demandNode.index, demandNode.value)

    # Second scenario
    demandNode = _node()
    demandNode.value = [0.1, 0.3]  # DemandProfiles[1][0:conf.Time]
    demandNode.index = 2
    EN.set_Demand(demandNode.index, demandNode.value)
#
#    # Several RES nodes
#    resInNode = _node()
#    for xr in range(conf.NoRES):
#        resInNode.value = 1#RESProfs[xr][0:conf.Time]
#        resInNode.index = xr+1
#        EN.set_RES(resInNode.index, resInNode.value)
#
#    # Several hydro nodes
#    hydroInNode = _node()
#    for xh in range(conf.NoHydro):
#        hydroInNode.value = 100
#        hydroInNode.index = xh+1
#        EN.set_Hydro(hydroInNode.index, hydroInNode.value)

    # one generator off
#    EN.set_GenCoFlag(1, False)
    # reduce capcity of the other
#    EN.set_GenCoFlag(2, 499)
    # Run integrated pyene
#    EN.set_Hydro(1, 30.34)
    mod = EN.run()
    # Get total energy generation
    Total_Generation = EN.get_AllGeneration(mod)
    print('Total generation: ', Total_Generation)
    # Replace all conventional generation with hydropower
    EN.set_Hydro(1, Total_Generation)
    # Run system again
    mod = EN.run()
    # Check conventional power generation
    Total_Conv_Generation = EN.get_AllGeneration(mod, 'Conv')
    print('Total conventional generation: ', Total_Conv_Generation)
    # Add even more hydropower
    Additional_hydro = 100
    EN.set_Hydro(1, Total_Generation+Additional_hydro)
    # Run the system again
    mod = EN.run()
    # Check that the water is spilled instead of used by the pumps
    Hydropower_Left = EN.get_AllHydro(mod)
    print('Hydropower left', Hydropower_Left)
#    print('Demand Curtailed', EN.get_AllDemandCurtailment(mod))
#    print('\n\nOF: ', mod.OF.expr())
    # Get demand curtailment as required hydropower inputs
    # Add hydropower
#    EN.set_Hydro(1, 30.3331)
#    print('Values withiun EM', EN.EM.Weight)
    # Run integrated pyene
#    mod = EN.run()

    # Print results
#    print('\n\nOF: ', mod.OF.expr())
#    EN.NM.offPrint()
#    EN.NM.Print['Generation'] = True
#    EN.NM.Print['Losses'] = True
#    EN.Print_ENSim(mod, EN.EM, EN.NM)

    # Collect unused hydro:
    print()
    hydroOutNode = _node()
    for xh in range(EN.EM.size['Vectors']):
        hydroOutNode.index = xh+1
        print('Hydro %d', hydroOutNode.index, ' left: ',
              EN.get_Hydro(mod, hydroOutNode.index), ' (',
              EN.get_HydroMarginal(mod, hydroOutNode.index), ')',
              EN.get_HydroFlag(mod, hydroOutNode.index))

    # Collect output of pumps
    print()
    pumpNode = _node()
    for xp in range(EN.NM.pumps['Number']):
        pumpNode.index = xp+1
        pumpNode.value = EN.get_Pump(mod, xp+1)
        print('Pump %d: %f' % (pumpNode.index, pumpNode.value))

    # Collect RES spill
    print()
    resOutNode = _node()
    for xp in range(EN.NM.RES['Number']):
        resOutNode.index = xp+1
        resOutNode.value = EN.get_RES(mod, resOutNode)
        print('RES %d: %f' % (resOutNode.index, resOutNode.value))

    # Collect curtailment per node
    print()
    curNode = _node()
    for xn in range(EN.NM.networkE.number_of_nodes()):
        curNode.bus = xn+1
        curNode.value = EN.get_DemandCurtailment(mod, curNode.bus)
        print('Dem %d: %f' % (curNode.bus, curNode.value))

    # Collect all curtailment
    print()
    curAll = _node()
    curAll.value = EN.get_AllDemandCurtailment(mod)
    print('Total curtailment:', curAll.value)


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
        self.Weights = None
