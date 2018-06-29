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
    def MaxHydroAllowance_rule(m, xv):
        '''Constraint for maximum hydropower allowance'''
        return m.vHydropowerAllowance[xv] <= m.MaxHydro[xv]

    def HydroAllowance_rule(m, xv):
        '''Constraint to link hydropower allowance'''
        return m.vHydropowerAllowance[xv] >= m.WInFull[1, xv]-m.WOutFull[1, xv]

    def ZeroHydroIn_rule(m, xn, xv):
        '''Constraint to link hydropower allowance'''
        return m.WInFull[xn, xv] == 0

    def HyroActualUse_rule(m, xv):
        '''Constraint for actual hydropower used'''
        return mod.vHydroUse[xv] == m.WInFull[1, xv]-m.WOutFull[1, xv]

    def AdjustPyeneMod(m, EN):
        '''Final modifications for pyene to work with the integrated model'''
        # Set unused water inputs to zero
        m.sN0 = [x for x in range(EN.EM.LL['NosBal']+1)]
        m.sN0.pop(1)
        m.ZeroHydroIn = Constraint(m.sN0, m.sVec, rule=ZeroHydroIn_rule)

        # Variables used by pyene for claculating the objective function
        m = EN._AddPyeneCons(EN.EM, EN.NM, m)
        EN.OFaux = EN._Calculate_OFaux(EN.EM, EN.NM)
        m.OFh = EN.h
        m.OFhGC = EN.hGC
        m.OFFea = EN.hFea
        m.OFpenalty = EN.Penalty
        m.OFpumps = EN.NM.pumps['Value']
        m.base = EN.NM.networkE.graph['baseMVA']
        m.OFhDL = EN.hDL
        m.OFweights = EN.NM.scenarios['Weights']
        m.OFaux = EN.OFaux

        return m

    def OF_rule(m):
        ''' Combined objective function '''
        return (m.WaterValue*sum(m.WInFull[1, xv] for xv in m.sVec) +
                sum((sum(sum(m.vGCost[m.OFhGC[xh]+xg, xt] for xg in m.sGen) +
                         sum(m.vFea[m.OFFea[xh]+xf, xt] for xf in m.sFea) *
                         m.OFpenalty for xt in m.sTim) -
                     sum(m.OFpumps[xdl]*m.base *
                         sum(m.vDL[m.OFhDL[xh]+xdl+1, xt] *
                             m.OFweights[xt]
                             for xt in m.sTim) for xdl in m.sDL)) *
                    m.OFaux[xh] for xh in m.OFh))

    print('Running test')

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
    # Selected network file
    conf.TreeFile = 'TestCase.json'
    conf.NetworkFile = 'case4.json'
    # Location of the json directory
    conf.json = conf.json = os.path.join(os.path.dirname(__file__), 'json')
    # Consider single time step
    conf.Time = 24  # Number of time steps
#    conf.Weights = [0.5, 1]

    # Add hydropower plant
    conf.NoHydro = 3  # Number of hydropower plants
    conf.Hydro = [1, 2, 3]  # Location of hydropower plants
    conf.HydroMax = [1000, 1000, 1000]  # Capacity of hydropower plants
    conf.HydroCost = [0.01, 0.01,  0.01]  # Cost of water

    # Pumps
    conf.NoPump = 2  # Number of pumps
    conf.Pump = [1, 2]  # Location of pumps
    conf.PumpMax = [1, 1]  # Capacity of the pumps
    conf.PumpVal = [0.001, 0.001]  # Value of pumped water

    # RES Generators
    conf.NoRES = 2  # Number of RES generators
    conf.RESMax = [100, 100]  # Capacity of hydro
    conf.RES = [1, 2]  # Bus of RES generators
    conf.Cost = [0, 0]  # Cost of RES

    # Enable curtailment
    conf.Feasibility = True

    # Initialize network model using the selected configuration
    EN.initialise(conf)
    # Create conditions for having demand curtailment
    EN.set_GenCoFlag(1, False)  # Switching one generator off
    EN.set_GenCoFlag(2, 499)  # Reducing capacity of the other generator

#    EN.set_Hydro(1, 400000)
#    mod = EN.run(mod)
#    Needed_hydro = EN.get_AllDemandCurtailment(mod)
#    print('Required hydro ',Needed_hydro)

    '''                        Fourth step
    Initialise pyomo sets, parameters, and variables
    '''
    mod = EN.build_Mod(EN.EM, EN.NM, mod)

    '''                         Fifth step
    Redefine hydropower inputs as variables
    '''
    del mod.WInFull
    if conf.NoHydro > 1:
        mod.WInFull = Var(mod.sNodz, mod.sVec, domain=NonNegativeReals,
                          initialize=0.0)
    else:
        mod.WInFull = Var(mod.sNodz, domain=NonNegativeReals, initialize=0.0)

    '''                         Sixth step                                  
    Define hydropower allowance
    Assuming pywr were to have these values in a variable called
    vHydropowerAllowance
    '''
    mod.vHydropowerAllowance = Var(mod.sVec, domain=NonNegativeReals,
                                   initialize=0.0)
    mod.MaxHydro = np.zeros(conf.NoHydro, dtype=float)
    for xh in range(conf.NoHydro):
        mod.MaxHydro[xh] = 134000
    # Add constraint to limit hydropower allowance
    mod.MaxHydroAllowance = Constraint(mod.sVec, rule=MaxHydroAllowance_rule)

    # Link hydropower constraint to pyene
    mod.HydroAllowance = Constraint(mod.sVec, rule=HydroAllowance_rule)

    '''                        Seventh step
    Make final modifications to pyene, so that the new constraints and
    objective function work correctly
    '''
    mod = AdjustPyeneMod(mod, EN)

    '''                        Eigth step
    Define a new objective function.
    Note that the if the value of hydropower:
    (i) hydro >= 10000 then the hydropower will never be used
    (ii) 1000 < hydro <= 270 then hydro will only avoid demand curtailment
    (iii) 270 < hydro <= 104 then Hydro will displace some generators
    (iv) 104 < hydro then Hydro will replace all other generation
    only be used for avoiding demand curtailment
    '''
    # Collect water use
    # Assuming that total hydropower use is assigned to a different variable
    mod.WaterValue = 104#9999
    mod.OF = Objective(rule=OF_rule, sense=minimize)

    '''                        Ninth step
    Running the model
    '''
    # Optimise
    opt = SolverFactory('glpk')
    # Print
    results = opt.solve(mod)



#    # Fake weather engine
#    FileName = 'TimeSeries.json'
#    (DemandProfiles, NoDemPeriod, BusDem, LinkDem, NoRES, NoRESP,
#     LLRESType, LLRESPeriod, RESProfs, RESBus, RESLink, NoLink,
#     Nohr) = EN.ReadTimeS(FileName)

#    # Single demand node (first scenario)
#    demandNode = _node()
#    demandNode.value = [0.2, 0.1]  # DemandProfiles[0][0:conf.Time]
#    demandNode.index = 1
#    EN.set_Demand(demandNode.index, demandNode.value)
#
#    # Second scenario
#    demandNode = _node()
#    demandNode.value = [0.1, 1]  # DemandProfiles[1][0:conf.Time]
#    demandNode.index = 2
#    EN.set_Demand(demandNode.index, demandNode.value)
#
#    # RES profile (first scenario)
#    resInNode = _node()
#    resInNode.value = np.zeros(conf.Time, dtype=float)
#    resInNode.index = 1
#    EN.set_RES(resInNode.index, resInNode.value)
#    resInNode.index = 2
#    EN.set_RES(resInNode.index, resInNode.value)
#
#    # RES profile (first scenario)
#    resInNode = _node()
#    resInNode.value = [0.5, 0.0]
#    resInNode.index = 2
#    EN.set_RES(resInNode.index, resInNode.value)
#    # COnstrain generation
#    EN.set_GenCoFlag(1, 249)
#    EN.set_GenCoFlag(2, 249)
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
#    EN.set_Hydro(1, 100)
#
#    mod = EN.run()
#    Demand_Curtailed = EN.get_AllDemandCurtailment(mod)
#    print('New demand curtailment ', Demand_Curtailed)

#    # Supply curtailed demand with hydro
#    EN.set_Hydro(1, Demand_Curtailed)

#    mod = EN.run()
#    Demand_Curtailed = EN.get_AllDemandCurtailment(mod)
#    print('New demand curtailment ', Demand_Curtailed)
#    mod = EN.run()
#    Demand_Curtailed = EN.get_AllDemandCurtailment(mod)
#    print('New demand curtailment ', Demand_Curtailed)
#    mod = EN.run()
#    Demand_Curtailed = EN.get_AllDemandCurtailment(mod)
#    print('New demand curtailment ', Demand_Curtailed)
#    mod = EN.run()
#    Demand_Curtailed = EN.get_AllDemandCurtailment(mod)
#    print('New demand curtailment ', Demand_Curtailed)
#    mod = EN.run()
#    Demand_Curtailed = EN.get_AllDemandCurtailment(mod)
#    print('New demand curtailment ', Demand_Curtailed)

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
#    EN.set_Hydro(1, Conv_Generation)
#    # Run again
#    mod = EN.run()
#    # Get new curtailment
#    New_Curtailed = EN.get_AllDemandCurtailment(mod)
#    print('New demand curtailment ', New_Curtailed)
#    # Get use of conventional generation
#    Use_ConvGeneration = EN.get_AllGeneration(mod, 'Conv')
#    print('Conventional generation ', Use_ConvGeneration)
#    # Fully cover conventional generation and demand with hydro
#    EN.set_Hydro(1, Conv_Generation+Demand_Curtailed)
#    # Run again
#    mod = EN.run()
#    # Get use of pumps
#    Final_Pump = EN.get_AllPumps(mod)
#    print('Energy used by pumps', Final_Pump)
#    # Get new curtailment
#    Final_Curtailed = EN.get_AllDemandCurtailment(mod)
#    print('New demand curtailment ', Final_Curtailed)
#    # Get use of conventional generation
#    Final_ConvGeneration = EN.get_AllGeneration(mod, 'Conv')
#    print('Conventional generation ', Final_ConvGeneration)
    
#    # Get total energy generation
#    Total_Generation = EN.get_AllGeneration(mod)
#    print('Total generation: ', Total_Generation)
#    # Replace all conventional generation with hydropower
#    EN.set_Hydro(1, Total_Generation)
#    # Run system again
#    mod = EN.run()
#    # Check conventional power generation
#    Total_Conv_Generation = EN.get_AllGeneration(mod, 'Conv')
#    print('Total conventional generation: ', Total_Conv_Generation)
#    # Add even more hydropower
#    Additional_hydro = 100
#    EN.set_Hydro(1, Total_Generation+Additional_hydro)
#    # Run the system again
#    mod = EN.run()
#    # Check that the water is spilled instead of used by the pumps
#    Hydropower_Left = EN.get_AllHydro(mod)
#    print('Hydropower left', Hydropower_Left)
#    print('Demand Curtailed', EN.get_AllDemandCurtailment(mod))
#    print('\n\nOF: ', mod.OF.expr())
    # Get demand curtailment as required hydropower inputs
    # Add hydropower
#    EN.set_Hydro(1, 30.3331)
#    print('Values withiun EM', EN.EM.Weight)
    # Run integrated pyene
#    mod = EN.run()

    # Print results
    print('\n\nOF: ', mod.OF.expr())
#    EN.NM.offPrint()
#    EN.NM.Print['Generation'] = True
#    EN.NM.Print['Losses'] = True
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
