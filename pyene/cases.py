""" This module contains the FutureDAMS - pyene test case definitions.

These test cases are used to assess the different functionalities of the
engine.

# TODO complete this description once the cases are written.

"""
from .engines.pyene import pyeneClass, pyeneConfig
from .fixtures import json_directory
from .engines.pyene import pyeneClass as pe
from pyomo.core import ConcreteModel
from pyomo.environ import SolverFactory
from .engines.pyeneO import PrintinScreen as PSc
import os
import json


def get_pyene(conf=None):
    """ Get pyene object."""

    return pyeneClass(conf)


def get_pyeneConfig():
    """ Get pyene object."""

    return pyeneConfig()


def test_pyeneAC(config):
    """ Run pypsa based AC power flow """
    config.NM.hydropower['Number'] = 0  # Number of hydropower plants
    
    # Create object
    EN = pyeneClass(config.EN)

    # Initialise with selected configuration
    EN.initialise(config)

    # Fake weather engine
    FileName = 'TimeSeries.json'
    (DemandProfiles, NoDemPeriod, BusDem, LinkDem, NoRES, NoRESP,
     LLRESType, LLRESPeriod, RESProfs, RESBus, RESLink, NoLink,
     Nohr) = EN.ReadTimeS(FileName)
    
    # Profiles
    demandNode = _node()
    demandNode.value = DemandProfiles[0][:]
    demandNode.index = 1
    EN.set_Demand(demandNode.index, demandNode.value)

    # Second scenario
    demandNode = _node()
    demandNode.value = DemandProfiles[1][:]
    demandNode.index = 2
    EN.set_Demand(demandNode.index, demandNode.value)
    
    EN_Interface = EN.getClassInterfaces()

    Model_Pypsa = [EN_Interface.pyene2pypsa(EN.NM, x)[0]
                   for x in range(config.NM.scenarios['NoDem'])]

    for xscen in range(config.NM.scenarios['NoDem']):
        Model_Pypsa[xscen].pf()

    EN.NM.print(Model_Pypsa, range(config.NM.scenarios['NoDem']), None, True)

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
    # Disable pyeneH
    conf.HM.settings['Flag'] = False

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

    if conf.EN.solverselection['pyomo']:
        # Run integrated pyene
        mod = ConcreteModel()
        mod = EN.run(mod)

        # Print results
        print('\n\nOF: ', mod.OF.expr())
        # EN.NM.offPrint()
        # EN.NM.Print['Generation'] = True
        # EN.NM.Print['Losses'] = True
        # EN.NM.Print['Flows'] = True
        # EN.NM.Print['GenBus'] = True
        EN.Print_ENSim(mod)

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

        # # Collect curtailment per node
        # print()
        # curNode = _node()
        # for xn in range(EN.NM.ENetwork.get_NoBus()):
        #     curNode.bus = xn+1
        #     curNode.value = EN.get_DemandCurtailment(mod, curNode.bus)
        #     print('Dem %d: %f' % (curNode.bus, curNode.value))

        # Collect all curtailment
        print()
        curAll = _node()
        curAll.value = EN.get_AllDemandCurtailment(mod)
        print('Total curtailment:', curAll.value)

    if conf.EN.solverselection['glpk']:
        # Energy model in glpk
        from .engines.pyene_Models import EnergyandNetwork as ENMod
        Model = ENMod(EN.EM, EN.NM, EN)
        Model.optimisationENM()
        # Print results
        print('\n\nOF: ', Model.GetObjectiveFunctionENM())
        Prints = PSc(EN)
        Prints.PrintallResults(Model)

        # Collect unused hydro:
        print()
        OutputsTree = Model.GetOutputsTree()
        EnergybalanceDual = Model.GetEnergybalanceDual()
        aux1 = Model.PenaltyCurtailment/Model.BaseUnitPower
        hydroOutNode = _node()
        for xv in range(Model.NumberTrees):
            hydroOutNode.index = xv+1
            if -EnergybalanceDual[xv, 1] > aux1:
                aux2 = True
            else:
                aux2 = False
            print('Hydro %d', hydroOutNode.index, ' left: ',
                  OutputsTree[xv, 1], ' (',
                  int(EnergybalanceDual[xv, 1]), ')',
                  aux2)

        # Collect output of pumps
        print()
        if Model.NumberPumps > 0:
            pumpNode = _node()
            PumpOperation = Model.GetPumpOperation()
            for xp in range(Model.NumberPumps):
                value = 0
                for xh in Model.LongTemporalConnections:
                    for xt in range(Model.ShortTemporalConnections):
                        value += PumpOperation[xh, xt, xp]
                pumpNode.index = xp+1
                pumpNode.value = value
                print('Pump %d: %f' % (pumpNode.index, pumpNode.value))

        # Collect RES spill
        print()
        if Model.NumberRESGen > 0:
            resOutNode = _node()
            RESGeneration = Model.GetRESGeneration()
            for xp in range(Model.NumberRESGen):
                value = 0
                for xh in Model.LongTemporalConnections:
                    acu = 0
                    for xt in range(Model.ShortTemporalConnections):
                        value += (Model.MaxRESGen[xp]*\
                            Model.RESScenarios[xh, xt, xp] \
                            * Model.BaseUnitPower -
                            RESGeneration[xh, xt, xp])
                resOutNode.index = xp+1
                resOutNode.value = value
                print('RES %d: %f' % (resOutNode.index, resOutNode.value))

        # # Collect curtailment per node
        # print()
        # curNode = _node()
        # for xn in range(EN.NM.ENetwork.get_NoBus()):
        #     curNode.bus = xn+1
        #     curNode.value = EN.get_DemandCurtailment(mod, curNode.bus)
        #     print('Dem %d: %f' % (curNode.bus, curNode.value))

        # Collect all curtailment
        print()
        curAll = _node()
        values = [0, 0]
        value = 0
        if Model.FlagProblem and Model.FlagFeasibility:
            LoadCurtailment = Model.GetLoadCurtailmentNodes()
        else:
            LoadCurtailment = Model.GetLoadCurtailmentSystemED()
        for xs in Model.LongTemporalConnections:
            for xt in range(Model.ShortTemporalConnections):
                if Model.FlagProblem and LoadCurtailment is not None:
                    for k in range(Model.NumberContingencies + 1):
                        for ii in range(Model.NumberNodesPS):
                            value += LoadCurtailment[xs, xt, k, ii]
                            values[EN.NM.ENetwork.Bus[ii].get_LT()] += \
                                LoadCurtailment[xs, xt, k, ii]
                if not Model.FlagProblem and LoadCurtailment is not None:
                    value += LoadCurtailment[xs, xt]
        curAll.value = value, values
        print('Total curtailment:', curAll.value)

def test_pyeneRES(conf):
    '''
    A case study with 2 PV generators located at two buses for 2 representative winter days
    '''

    # Initialise simulation methods
    conf.HM.settings['Flag'] = False
    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True
    conf.NM.settings['NoGenerators'] = 12
    #conf.NM.settings['File'] = os.path.join(json_directory(), 'caseGhana_Sim40_BSec_ManualV02.json')
    conf.NM.settings['NoTime'] = 24
    conf.NM.scenarios['Weights'] = [1 for _ in range(conf.NM.settings['NoTime'])]
    #conf.NM.scenarios['Number'] = 4  # Four scenarios including winter weekday/weekend and summer weekday/weekend
    conf.NM.scenarios['NoDem'] = 2  # Four demand profiles (repeated once) including winter weekday/weekend and summer weekday/weekend

    # RES generators
    conf.NM.RES['Number'] = 2  # Number of RES generators
    conf.NM.RES['Bus'] = [1, 4]  # Location (bus) of generators
    conf.NM.RES['Max'] = [10, 15]  # Generation capacity
    conf.NM.RES['Cost'] = [0.001, 0.001]  # Costs

    # Hydro generators
    conf.NM.hydropower['Number'] = 1  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [5]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [200]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.001]  # Costs

    # Create object
    EN = pe(conf.EN)

    # Initialise with selected configuration
    EN.initialise(conf)

    ################### case 1 - Winter scenarios ###########################

    # Demand profile
    fileName = os.path.join(json_directory(), 'TimeSeries.json')
    Eprofiles = json.load(open(fileName))
    EN.set_Demand(1, Eprofiles['Demand']['Values'][0])  # First scenario demand profile - Winter Weekday
    if conf.NM.scenarios['NoDem'] == 2:
        EN.set_Demand(2, Eprofiles['Demand']['Values'][1])  # Second scenario demand profile - Winter Weekend


    # RES profile
    resInNode = _node()
    for xr in range(EN.NM.RES['Number']):
        resInNode.value = Eprofiles['PV']['Values'][xr]  # Winter profile for RES generator 1 & 2
        resInNode.index = xr + 1
        EN.set_RES(resInNode.index, resInNode.value)

    # Hydro profile
    hydroInNode = _node()
    for xh in range(EN.NM.hydropower['Number']):
        hydroInNode.value = 1000  # Winter dry seasons MWh
        hydroInNode.index = xh + 1
        EN.set_Hydro(hydroInNode.index, hydroInNode.value)

    # Solve the model and generate results
    m = ConcreteModel()
    m = EN.run(m)
    EN.Print_ENSim(m)
    print('Total curtailment:', EN.get_AllDemandCurtailment(m))
    print('Spill ', EN.get_AllRES(m))
    print('OF   : ', m.OF.expr())

    ################### case 2 - Summer scenarios ###########################

    # Demand profile
    EN.set_Demand(1, Eprofiles['Demand']['Values'][4])  # Third scenario demand profile - Summer Weekday
    if conf.NM.scenarios['NoDem'] == 2:
        EN.set_Demand(2, Eprofiles['Demand']['Values'][5])  # Fourth scenario demand profile - Summer Weekend

    # RES profile
    for xr in range(EN.NM.RES['Number']):
        resInNode.value = Eprofiles['PV']['Values'][xr+4]  # Summer profile for RES generator 1 & 2
        resInNode.index = xr + 1
        EN.set_RES(resInNode.index, resInNode.value)

    # Hydro profile
    for xh in range(EN.NM.hydropower['Number']):
        hydroInNode.value = 10000  # Summer rainy seasons MWh
        hydroInNode.index = xh + 1
        EN.set_Hydro(hydroInNode.index, hydroInNode.value)

    # Solve the model and generate results
    m = ConcreteModel()
    m = EN.run(m)
    EN.Print_ENSim(m)
    print('Total curtailment:', EN.get_AllDemandCurtailment(m))
    print('Spill ', EN.get_AllRES(m))
    print('OF   : ', m.OF.expr())

def hydro_example_tobeerased(conf):
    """ Execute pyene to run the example of baseload - THIS NEEDS TO BE ERASED \
        IN A FUTURE RELEASE"""
    # Disable pyeneH
    conf.HM.settings['Flag'] = False

    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True
    conf.NM.settings['NoTime'] = 24  # Single period
    conf.NM.scenarios['Weights'] = [1 for _ in range(24)]  # Add weights 
        # to the time steps

    # Adding hydropower plants
    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 5]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [200, 200]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.001, 0.001]  # Costs

    # Create object
    EN = pyeneClass(conf.EN)

    # Initialise with selected configuration
    EN.initialise(conf)

    # Single demand node (first scenario)
    demandNode = _node()
    demandNode.value = [0.3333, 0.2623, 0.2350, 0.2240, 0.2240, 
        0.2514, 0.3825, 0.6284, 0.6503, 0.5574, 0.5301, 0.5137, 
        0.5355, 0.5027, 0.4918, 0.5464, 0.7760, 0.9891, 1.0000, 
        0.9399, 0.8634, 0.8142, 0.6885, 0.4918]
    demandNode.index = 1
    EN.set_Demand(demandNode.index, demandNode.value)

    # Several hydro nodes
    hydroInNode = _node()
    for xh in range(EN.NM.hydropower['Number']):
        hydroInNode.value = 10000
        hydroInNode.index = xh+1
        EN.set_Hydro(hydroInNode.index, hydroInNode.value)
    
    # Run integrated pyene
    m = ConcreteModel()
    m = EN.run(m)
    EN.Print_ENSim(m)
    print('Total curtailment:', EN.get_AllDemandCurtailment(m))
    print('Spill ', EN.get_AllRES(m))
    print('OF   : ', m.OF.expr())


def test_pyenetest(mthd):
    '''Test specific functionalities'''
    from .engines.pyeneT import TestClass

    txt = 'test' + str(mthd)
    method_to_call = getattr(TestClass, txt)
    method_to_call(TestClass)
