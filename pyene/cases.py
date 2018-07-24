""" This module contains the FutureDAMS - pyene test case definitions.

These test cases are used to assess the different functionalities of the
engine.

# TODO complete this description once the cases are written.

"""
from .engines.pyene import pyeneClass as pe, pyeneHDF5Settings as peHDF5, \
    RESprofiles as peRES
import numpy as np
import os
from pyomo.environ import *
from pyomo.core import *
from pyomo.opt import SolverFactory
from tables import *
import matplotlib.pyplot as plt

# pyene simulation test
def pyene():
    """ Get pyene object."""

    return pe()


def pyeneH5():
    '''Get class to build H5 files'''

    return peHDF5()


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
    mod = ConcreteModel()
    mod = EN.run(mod)

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


def test_pyenetest():
    '''Test specific functionalities'''
    print('Running test')
    res = peRES()
    sW = 27
    wind = np.zeros(sW, dtype=float)
    for xt in range(sW):
        wind[xt] = xt
    wind = res.buildWind(wind)
    print(wind)
#    plt.plot(wind)
#    plt.show()
    
    PVdirect = [0, 0, 0, 0.0998, 0.3149, 9.7061, 43.8524, 111.7508, 191.1023,
                247.9384, 258.0696, 269.3670, 274.4125, 269.6220, 244.3720,
                200.6101, 130.3558, 57.1130, 14.2986, 1.4752, 0.3625, 0, 0, 0]
    PVdiffuse = [0, 0, 0, 0, 6.4500, 27.4982, 66.7002, 105.7961, 139.0234,
                 165.9763, 184.6349, 195.5550, 196.7047, 187.8306, 169.0823,
                 143.2314, 110.5894, 72.1680, 34.3754, 9.4169, 0, 0, 0, 0]
    res.setPVday()
    pv = res.buildPV(PVdirect, PVdiffuse)
    print(pv)
#    plt.plot(pv)
#    plt.show()



    aux[1000]
    '''                         First step
    create pyomo model
    '''
    mod = ConcreteModel()

    '''                         Second step
    create pyene object
    '''
    EN = pe()

    '''                          Third step
    initialise pyene with predefined configuration
    '''
    conf = default_conf()
    conf.json = os.path.join(os.path.dirname(__file__), 'json')
    # Selected network file
    conf.NetworkFile = 'case4.json'
    # Location of the json directory
    conf.json = conf.json = os.path.join(os.path.dirname(__file__), 'json')
    # Consider single time step
    conf.Time = 2  # Number of time steps
    conf.Weights = [0.5, 1]
    # Add hydropower plant
    conf.NoHydro = 1  # Number of hydropower plants
    conf.Hydro = [1]  # Location (bus) of hydro
    conf.HydroMax = [500]  # Generation capacity
    conf.HydroCost = [0.01]  # Costs
    # Pumps
    conf.NoPump = 1  # Number of pumps
    conf.Pump = [2]  # Location (bus) of pumps
    conf.PumpMax = [100]  # Generation capacity
    conf.PumpVal = [0.001]  # Value/Profit
    # RES generators
    conf.NoRES = 1  # Number of RES generators
    conf.RES = [3]  # Location (bus) of pumps
    conf.RESMax = [100]  # Generation capacity
    conf.Cost = [0.0001]  # Costs
    # Enable curtailment
    conf.Feasibility = True
    # Get Pyene model
    EN = pe()
    # Initialize network model using the selected configuration
    EN.initialise(conf)
    # Single demand node (first scenario)
#    demandNode = _node()
#    demandNode.value = [0.2, 0.1]  # DemandProfiles[0][0:conf.Time]
#    demandNode.index = 1
#    EN.set_Demand(demandNode.index, demandNode.value)
#    # Second scenario
#    demandNode = _node()
#    demandNode.value = [0.1, 1]  # DemandProfiles[1][0:conf.Time]
#    demandNode.index = 2
#    EN.set_Demand(demandNode.index, demandNode.value)
    # RES profile (first scenario)
    resInNode = _node()
    resInNode.value = [0.5, 6.0]
    resInNode.index = 1
    EN.set_RES(resInNode.index, resInNode.value)
    # RES profile (first scenario)
#    resInNode = _node()
#    resInNode.value = [0.5, 0.0]
#    resInNode.index = 2
#    EN.set_RES(resInNode.index, resInNode.value)
    # COnstrain generation
    EN.set_GenCoFlag(1, 200)
    EN.set_GenCoFlag(2, 200)
#    mod = ConcreteModel()
#    mod = EN.run(mod)
#    # Get RES spilled
#    RES_Spilled = EN.get_AllRES(mod)
#    print('RES spilled ', RES_Spilled)
#    # Get use of pumps
#    Pumps_Use = EN.get_AllPumps(mod)
#    print('Energy used by pumps', Pumps_Use)
#    # Get demand curtailed
#    Demand_Curtailed = EN.get_AllDemandCurtailment(mod)
#    print('Demand curtailed', Demand_Curtailed)
#    # Add hydro to replace conventional generation
#    Conv_Generation = EN.get_AllGeneration(mod, 'Conv')
#    print('Conventional generation ', Conv_Generation)
#    EN.set_Hydro(1, Conv_Generation)
#    # Run again
#    mod = ConcreteModel()
#    mod = EN.run(mod)
#    # Get new curtailment
#    New_Curtailed = EN.get_AllDemandCurtailment(mod)
#    print('New demand curtailment ', New_Curtailed)
#    # Get use of conventional generation
#    Use_ConvGeneration = EN.get_AllGeneration(mod, 'Conv')
#    print('Conventional generation ', Use_ConvGeneration)
#    # Fully cover conventional generation and demand with hydro
#    EN.set_Hydro(1, Conv_Generation+Demand_Curtailed)
    # Run again
    
    EN.set_LineCapacity(3, 80)
    
    ENH5 = peHDF5()
    pyenefileName = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'pyeneOutputs.h5')
#    print(pyenefileName)
#    aux[1000]
    ENH5.fileh = open_file(pyenefileName, mode='w')
    ENH5.SaveSettings(ENH5.fileh, EN, conf, ENH5.fileh.root)
    mod = ConcreteModel()
    mod = EN.run(mod)
    ENH5.saveResults(ENH5.fileh, EN, mod, ENH5.fileh.root, 0)
    ENH5.fileh.close()
    # Get use of pumps
    Final_Pump = EN.get_AllPumps(mod)
    print('Energy used by pumps', Final_Pump)
    # Get new curtailment
    Final_Curtailed = EN.get_AllDemandCurtailment(mod)
    print('New demand curtailment ', Final_Curtailed)
    # Get use of conventional generation
    Final_ConvGeneration = EN.get_AllGeneration(mod, 'Conv')
    print('Conventional generation ', Final_ConvGeneration)

    # Print results    
#    EN.NM.offPrint()
#    EN.NM.Print['Generation'] = True
#    EN.NM.Print['Losses'] = True
    print('\n\nOF: ', mod.OF.expr())
    EN.Print_ENSim(mod, EN.EM, EN.NM)
#    print(type(mod.WInFull))
#    if type(mod.WInFull) is np.ndarray:
#        print('Numpy array')
#        print(mod.WInFull)
#    else:
#        print('Pyomo class')
#        print(mod.WInFull)
            
    
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
        resOutNode.value = EN.get_RES(mod, resOutNode.index)
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

    # Consider demand, pumps, losses, curtailment, spill
    auxFlag = [True, False, False, False, False]
    value = EN.get_NetDemand(mod, auxFlag)
    print(value)
    
    auxFlags = [True, True, True, True, True]
    value = EN.get_OFparts(mod, auxFlags)
    print(value)
    
    print('Spill ', EN.get_AllRES(mod))



