""" This module contains the FutureDAMS - pyene test case definitions.

These test cases are used to assess the different functionalities of the
engine.

# TODO complete this description once the cases are written.

"""
from .engines.pyene import pyeneClass, pyeneConfig
from pyomo.core import ConcreteModel
from pyomo.environ import SolverFactory


def get_pyene(conf=None):
    """ Get pyene object."""

    return pyeneClass(conf)


def get_pyeneConfig():
    """ Get pyene object."""

    return pyeneConfig()


def test_pyeneE(config):
    """ Execute pyene to access pyeneE - Full json based simulation."""
    EN = pyeneClass(config.EN)
    (EM, EModel, results) = EN.ESim(config)
    EM.print(EModel)


# Network simulation test
def test_pyeneN(config):
    """ Execute pyene to access pyeneN - Full json based simulation."""
    # Create object
    EN = pyeneClass(config.EN)
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
    EN = pyeneClass(conf.EN)

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
    for xr in range(EN.NM.RES['Number']):
        resInNode.value = RESProfs[xr][:]
        resInNode.index = xr+1
        EN.set_RES(resInNode.index, resInNode.value)

    # Several hydro nodes
    hydroInNode = _node()
    for xh in range(EN.NM.hydropower['Number']):
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
    
    
    # Get default configuration
    from .engines.pyene import pyeneConfig
    import os
    jsonDirectory1 = os.path.join(os.path.dirname(__file__), 'json')
#    jsonDirectory2 = os.path.join(os.path.dirname(__file__), '..', 'tests',
#                                  'json')
#    jsonDirectory3 = os.path.join(os.path.dirname(__file__), '..', '..',
#                                  'test-case-integrated', 'fdtc_integrated',
#                                  'json')
    conf = pyeneConfig()
    EN = pyeneClass(conf.EN)
    (HM, HModel, results)=EN.HSim(conf)
    HM.print(HModel)
    print('OF: ', HModel.OF.expr())
    print()
    print(HModel.vHout[0, 0].value)
    print(HModel.vHout[0, 1].value)
    print(HModel.vHout[0, 2].value)
    print(HModel.vHout[1, 2].value)
    print(HModel.vHout[1, 3].value)
    print(HModel.vHout[1, 4].value)
    print(HModel.vHout[2, 2].value)
    print(HModel.vHout[2, 3].value)
    print(HModel.vHout[2, 4].value)
    print()
    
    
    
    
    
    
    
    
    
#    HM.print_settings(HModel)
    
    
#    from .engines.pyeneH import HydrologyClass
#
#    HM = HydrologyClass(conf.HM)
#    HM.initialise()
#
#    mod = ConcreteModel()
#    mod = HM.addSets(mod)
#    mod = HM.addPar(mod)
#    mod = HM.addVars(mod)
#    mod = HM.addCon(mod)
#    from pyomo.core import Objective, minimize
#    mod.OF = Objective(rule=HM.OF_rule, sense=minimize)
#    opt = SolverFactory('glpk')
#    results = opt.solve(mod)
##    print(results)
#    HM.print(mod)
#    print('OF: ', mod.OF.expr())
#    HM.print_settings(mod)
    
#    # Example of the contents of conf
#    conf = pyeneConfig()
#    conf.EM.settings['File'] = \
#        os.path.join(jsonDirectory3, 'ResolutionTreeMonth01.json')
##    conf.NM.settings['File'] = \
##        os.path.join(jsonDirectory3, 'case4modified.json')
#    conf.NM.settings['File'] = \
#        os.path.join(jsonDirectory3, 'case14.json')
#
#    # Hydropower
#    conf.NM.hydropower['Number'] = 3  # Number of hydropower plants
#    conf.NM.hydropower['Bus'] = [2, 3, 4]  # Location (bus) of hydro
#    conf.NM.hydropower['Max'] = [1000, 1000, 1000]  # Generation capacity
#    conf.NM.hydropower['Cost'] = [0.01, 0.01,  0.01]  # Costs
#
#    # Pumps
#    conf.NM.pumps['Number'] = 4  # Number of pumps
#    conf.NM.pumps['Bus'] = [1, 2, 3, 4]  # Location (bus) of pumps
#    conf.NM.pumps['Max'] = [100, 100, 100, 100]  # Generation capacity
#    conf.NM.pumps['Value'] = [0.001, 0.001, 0.001, 0.001]  # Value/Profit
#
#    # RES generators
#    conf.NM.RES['Number'] = 4  # Number of RES generators
#    conf.NM.RES['Bus'] = [1, 2, 3, 4]  # Location (bus) of pumps
#    conf.NM.RES['Max'] = [1000, 1000, 1000, 1000]  # Generation capacity
#    conf.NM.RES['Cost'] = [0, 0, 0, 0]  # Costs
#
#    # Network considerations
#    conf.NM.settings['Security'] = []  # List of contingescies to test
#    conf.NM.settings['Losses'] = True  # Model losses
#    conf.NM.settings['Feasibility'] = True  # Add dummy generators
#    conf.NM.settings['NoTime'] = 24  # Number of time steps
##    conf.NM.scenarios['Weights'] = 1  # Weights for each time step
#
#    # Scenarios
#    conf.NM.scenarios['NoDem'] = 2  # Number of demand profiles
#    conf.NM.scenarios['NoRES'] = 1  # Number of RES profiles
##    conf.NM.settings['Pieces'] = [1]
#
#    # Create object
#    EN = pyeneClass(conf.EN)
#
#    # Initialise with selected configuration
#    EN.initialise(conf)
#
#    # Add new profiles
#    import json
#    fileName = os.path.join(jsonDirectory1, 'UKElectricityProfiles.json')
#    Eprofiles = json.load(open(fileName))
#
#    EN.set_Demand(1, Eprofiles['Winter']['Weekday'])
#    EN.set_Demand(2, Eprofiles['Winter']['Weekend'])
#
#    # Introduce network issues
##    EN.set_GenCoFlag(1, 450)
##    EN.set_GenCoFlag(2, 450)
#    for xb in range(EN.NM.networkE.number_of_edges()):
#        EN.set_LineCapacity(xb, 20)
#
##    EN.set_Hydro(1, 1108.00167264)
#    import numpy
#    EN.set_RES(1, numpy.ones(24, dtype=int))
#
#    # Save simulation settings
#    from tables import open_file
#    pyenefileName = os.path.join(os.path.dirname(__file__), '..',
#                                 'outputs', 'pyeneOutputs.h5')
#    print(pyenefileName)
#    ENH5 = EN.getClassOutputs()
#    ENH5.fileh = open_file(pyenefileName, mode='w')
#    ENH5.SaveSettings(ENH5.fileh, EN, ENH5.fileh.root)
#
#    # Run integrated pyene
#    mod = ConcreteModel()
#    mod = EN.run(mod)
#    EN.Print_ENSim(mod, EN.EM, EN.NM)
#
#    # Save results
#    ENH5.saveResults(ENH5.fileh, EN, mod, ENH5.fileh.root, 1)
#
#    print(mod.OF.expr())
#    print(EN.EM.data)
#
#    # Print results
##    EN.NM.offPrint()
##    EN.NM.Print['Generation'] = True
##    EN.NM.Print['Losses'] = True
#    print('\n\nOF: ', mod.OF.expr())
#    EN.Print_ENSim(mod, EN.EM, EN.NM)
#    import numpy as np
#    if type(mod.WInFull) is np.ndarray:
#        print('Numpy array')
#        print(mod.WInFull)
#    else:
#        print('Pyomo class')
#        print(mod.WInFull)
#    xt = 0
#    xs = 0
#    print(EN.get_AllLoss(mod, 'snapshot', times=[xt], scens=[xs]))
#
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
#    for xh in range(EN.NM.hydropower['Number']):
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
#    print('OF: ', EN.get_OFparts(mod, [True, True, True, True, True]))



