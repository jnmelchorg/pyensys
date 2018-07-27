""" This module contains the FutureDAMS - pyene test case definitions.

These test cases are used to assess the different functionalities of the
engine.

# TODO complete this description once the cases are written.

"""
from .engines.pyene import pyeneClass as pe
import numpy as np
import os
from pyomo.environ import *
from pyomo.core import *
from pyomo.opt import SolverFactory
from tables import *
from pyene.engines.pyene import pyeneConfig
import json

# pyene simulation test
def pyene():
    """ Get pyene object."""

    return pe()

class default_conf(pyeneConfig):
    def __init__(self):
        obj = pyeneConfig()
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

        self.EM.fRes = os.path.join(os.path.dirname(__file__), '..', 'tests', 'json', 'ResolutionTreeMonth01.json')
    
        self.init = False  # skip file reading?
        self.TreeFile = 'ResolutionTreeMonth01.json'  # Selected tree file
        self.NetworkFile = 'case4.json'  # Selected network file
        self.json = os.path.join(os.path.dirname(__file__), '..', 'tests', 'json')  # Location of the json directory

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
    EN = pe(config.EN)
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
    EN = pe(conf.EN)

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
    conf = default_conf()
    print('\n\n', os.path.dirname(__file__))

    conf.NetworkFile = 'case4.json'
    conf.TreeFile = 'ResolutionTreeMonth01.json'
    conf.EM.fRes = os.path.join(os.path.dirname(__file__), '..', 'tests', 'json', 'ResolutionTreeMonth01.json')
    conf.Time = 1  # Single period
    # Create object
    EN = pe(conf.EN)
    # Initialise with selected configuration
    EN.initialise(conf)
    # Run integrated pyene
    mod = ConcreteModel()
    mod = EN.run(mod)
    EN.Print_ENSim(mod, EN.EM, EN.NM)
    print(mod.OF.expr())
    
    

#    '''                         First step
#    create pyomo model
#    '''
#    mod = ConcreteModel()
#
#    '''                         Second step
#    create pyene object
#    '''
#    EN = pe()
#
#    '''                          Third step
#    initialise pyene with predefined configuration
#    '''
#    conf = default_conf()
#    # Location of the json directory
#    conf.json = os.path.join(os.path.dirname(__file__), 'json')
#    # Selected network file
#    conf.NetworkFile = 'case4.json'
#    # Consider single time step
#    conf.Time = 24  # Number of time steps
#    conf.Weights = np.ones(conf.Time, dtype=int)
#    # Add hydropower plant
#    conf.NoHydro = 3  # Number of hydropower plants
#    conf.Hydro = [4, 3, 2]  # Location (bus) of hydro
#    conf.HydroMax = [1000, 1000, 1000]  # Generation capacity
#    conf.HydroCost = [0.01, 0.01, 0.01]  # Costs
#    # Pumps
#    conf.NoPump = 3  # Number of pumps
#    conf.Pump = [2, 3, 4]  # Location (bus) of pumps
#    conf.PumpMax = [100, 100, 100]  # Generation capacity
#    conf.PumpVal = [0.001, 0.001, 0.001]  # Value/Profit
#    # RES generators
#    conf.NoRES = 1  # Number of RES generators
#    conf.RES = [1]  # Location (bus) of pumps
#    conf.RESMax = [100]  # Generation capacity
#    conf.Cost = [0]  # Costs
#    # Enable curtailment
#    conf.Feasibility = True
#    # Enable losses
#    conf.Losses = True
#    conf.NoRESProfiles = 1
#    # Get Pyene model
#    EN = pe()
#    # Initialize network model using the selected configuration
#    conf.GenPieces = 100
##    conf.NM_RES_Number = 1
##    conf.NM.settings['Notime']=1
##    
#    
#    EN.initialise(conf)
#
#    fileName = os.path.join(conf.json, 'UKElectricityProfiles.json')
#    Eprofiles = json.load(open(fileName))
##    print(Eprofiles)
#    EN.set_Demand(1, Eprofiles['Winter']['Weekday'])
#    EN.set_Demand(2, Eprofiles['Winter']['Weekend'])
##    EN.set_GenCoFlag(2, 500)
##    EN.set_GenCoFlag(1, False)
#    # Introducing network constraints
#    for xb in range(EN.NM.networkE.number_of_edges()):
#        EN.set_LineCapacity(xb, 50)
#    EN.set_Hydro(1, 580.43)
##    aux[1000]
#
##    # Single demand node (first scenario)
##    demandNode = _node()
##    demandNode.value = [0.2, 0.1]  # DemandProfiles[0][0:conf.Time]
##    demandNode.index = 1
##    EN.set_Demand(demandNode.index, demandNode.value)
##    # Second scenario
##    demandNode = _node()
##    demandNode.value = [0.1, 1]  # DemandProfiles[1][0:conf.Time]
##    demandNode.index = 2
##    EN.set_Demand(demandNode.index, demandNode.value)
##    # RES profile (first scenario)
##    resInNode = _node()
##    resInNode.value = [0.5, 6.0]
##    resInNode.index = 1
##    EN.set_RES(resInNode.index, resInNode.value)
##    # RES profile (first scenario)
##    resInNode = _node()
##    resInNode.value = [0.5, 0.0]
##    resInNode.index = 2
##    EN.set_RES(resInNode.index, resInNode.value)
##    # COnstrain generation
##    EN.set_GenCoFlag(1, 200)
##    EN.set_GenCoFlag(2, 200)
#
#    pyenefileName = os.path.join(os.path.dirname(__file__), '..',
#                                 'outputs', 'pyeneOutputs.h5')
#    print(pyenefileName)
#    ENH5 = EN.getClassOutputs()
#    ENH5.fileh = open_file(pyenefileName, mode='w')
#    ENH5.SaveSettings(ENH5.fileh, EN, conf, ENH5.fileh.root)
#
#
#    mod = ConcreteModel()
#    mod = EN.run(mod)
#
#    ENH5.saveResults(ENH5.fileh, EN, mod, ENH5.fileh.root, 1)
#    ENH5.fileh.close()
#    # Print results    
##    EN.NM.offPrint()
##    EN.NM.Print['Generation'] = True
##    EN.NM.Print['Losses'] = True
#    print('\n\nOF: ', mod.OF.expr())
#    EN.Print_ENSim(mod, EN.EM, EN.NM)
#    if type(mod.WInFull) is np.ndarray:
#        print('Numpy array')
#        print(mod.WInFull)
#    else:
#        print('Pyomo class')
#        print(mod.WInFull)
#    xt = 0
#    xs = 0
#    print(EN.get_AllLoss(mod, 'snapshot', times=[xt], scens=[xs]))
##    
#    # Collect unused hydro:
#    print()
#    hydroOutNode = _node()
#    for xh in range(EN.EM.size['Vectors']):
#        hydroOutNode.index = xh+1
#        print('Hydro %d', hydroOutNode.index, ' left: ',
#              EN.get_Hydro(mod, hydroOutNode.index), ' (',
#              EN.get_HydroMarginal(mod, hydroOutNode.index), ')',
#              EN.get_HydroFlag(mod, hydroOutNode.index))
#
#    # Collect output of pumps
#    print()
#    pumpNode = _node()
#    for xp in range(EN.NM.pumps['Number']):
#        pumpNode.index = xp+1
#        pumpNode.value = EN.get_Pump(mod, xp+1)
#        print('Pump %d: %f' % (pumpNode.index, pumpNode.value))
#
#    # Collect RES spill
#    print()
#    resOutNode = _node()
#    for xp in range(EN.NM.RES['Number']):
#        resOutNode.index = xp+1
#        resOutNode.value = EN.get_RES(mod, resOutNode.index)
#        print('RES %d: %f' % (resOutNode.index, resOutNode.value))
#
#    # Collect curtailment per node
#    print()
#    curNode = _node()
#    for xn in range(EN.NM.networkE.number_of_nodes()):
#        curNode.bus = xn+1
#        curNode.value = EN.get_DemandCurtailment(mod, curNode.bus)
#        print('Dem %d: %f' % (curNode.bus, curNode.value))
#
#    # Collect all curtailment
#    print()
#    curAll = _node()
#    curAll.value = EN.get_AllDemandCurtailment(mod)
#    print('Total curtailment:', curAll.value)
#
#    # Collect marginal costs
#    print()
#    for xh in range(conf.NoHydro):
#        print('MHyd ',xh+1, ': ', EN.get_HydroMarginal(mod, xh+1))
##
##    # Consider demand, pumps, losses, curtailment, spill
##    auxFlag = [True, False, False, False, False]
##    value = EN.get_NetDemand(mod, auxFlag)
##    print(value)
##    
##    auxFlags = [True, True, True, True, True]
##    value = EN.get_OFparts(mod, auxFlags)
##    print(value)
##    
#    print('Spill ', EN.get_AllRES(mod))
#    print('OF: ', mod.OF.expr())



