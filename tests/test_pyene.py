""" Test the pyeneE engine. """
from fixtures import testConfig, json_directory
from pyene.engines.pyene import pyeneClass as pe
import numpy as np
import os
from pyomo.core import ConcreteModel, Constraint, Var, NonNegativeReals, \
                       Objective, minimize
from pyomo.environ import SolverFactory


# Interaction node
class _node():
    def __init__(self):
        self.value = None
        self.index = None
        self.bus = None
        self.marginal = None
        self.flag = False


# Energy balance and network simulation
def test_pyene_Small():
    print('test_pyene_Small')
    conf = testConfig()
#    conf.TreeFile = 'ResolutionTreeMonth01.json'
    conf.EM.settings['File'] = os.path.join(json_directory(),
                                            'ResolutionTreeMonth01.json')
    conf.NM.settings['NoTime'] = 1  # Number of time steps
    # Create object
    EN = pe(conf.EN)
    # Initialise with selected configuration
    EN.initialise(conf)
    # Run integrated pyene
    mod = ConcreteModel()
    mod = EN.run(mod)
    EN.Print_ENSim(mod, EN.EM, EN.NM)
    print(mod.OF.expr())

    assert 0.0001 >= abs(mod.OF.expr()-21952.5*7*4.25)


# Adding hydropower
def test_pyene_SmallHydro():
    print('test_pyene_SmallHydro')
    conf = testConfig()
    conf.NetworkFile = 'case4.json'
    conf.EM.settings['File'] = os.path.join(json_directory(),
                                            'ResolutionTreeMonth01.json')
    conf.NM.settings['NoTime'] = 1  # Single period

    # Adding hydropower plants
    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [100, 100]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.01, 0.01]  # Costs

    # Create object
    EN = pe(conf.EN)
    # Initialise with selected configuration
    EN.initialise(conf)
    # Add hydro nodes
    for xh in range(conf.NM.hydropower['Number']):
        EN.set_Hydro(xh+1, 1000)
    # Run integrated pyene
    mod = ConcreteModel()
    mod = EN.run(mod)
    EN.Print_ENSim(mod, EN.EM, EN.NM)
    print('\n%f ' % mod.OF.expr())

    assert 0.0001 >= abs(mod.OF.expr()-527048.8750)


# Converting to pypsa
def test_pyene2pypsa():
    print('test_pyene2pypsa')
    conf = testConfig()
    # Selected network file
    conf.NM.settings['File'] = os.path.join(json_directory(), 'case14.json')
    # Define number of time spets
    conf.NM.settings['NoTime'] = 1  # Single period
    # Hydropower
    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [100, 100]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.01, 0.01]  # Costs

    # Pumps
    conf.NM.pumps['Number'] = 1  # Number of pumps
    conf.NM.pumps['Bus'] = [3]  # Location (bus) of pumps
    conf.NM.pumps['Max'] = [1000]  # Generation capacity
    conf.NM.pumps['Value'] = [0.001]  # Value/Profit

    # RES generators
    conf.NM.RES['Number'] = 3  # Number of RES generators
    conf.NM.RES['Bus'] = [3, 4, 5]  # Location (bus) of pumps
    conf.NM.RES['Max'] = [500, 500, 500]  # Generation capacity
    conf.NM.RES['Cost'] = [0.0001, 0.0001, 0.0001]  # Costs

    # Get Pyene model
    EN = pe(conf.EN)
    # Initialize network model using the selected configuration
    EN.initialise(conf)
    # Convert to pypsa
    xscen = 0  # Selected scenario
    P2P = EN.getClassInterfaces()
    (nu, pypsaFlag) = P2P.pyene2pypsa(EN.NM, xscen)
    # Run pypsa
    nu.pf()

    assert 0.0001 >= abs(nu.lines_t.p0['Line1'][0] - 158.093958)


# Test iteration where hydro covers load curtailment
def test_pyene_Curtailment2Hydro():
    '''
    Identify demand curtailment in a first iteration
    and supply customers with hydropower in another.
    '''
    print('test_pyene_Curtailment2Hydro')
    conf = testConfig()
    # Consider single time step
    conf.NM.settings['NoTime'] = 1  # Single period
    # Add hydropower plant
    conf.NM.hydropower['Number'] = 1  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [100]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.01]  # Costs
    # Enable curtailment
    conf.NM.settings['Feasibility'] = True
    # Get Pyene model
    EN = pe(conf.EN)
    # Initialize network model using the selected configuration
    EN.initialise(conf)
    # one generator off
    EN.set_GenCoFlag(1, False)
    # reduce capcity of the other generator
    EN.set_GenCoFlag(2, 499)
    # Run integrated pyene
    mod = ConcreteModel()
    mod = EN.run(mod)
    print('Marginal ', EN.get_HydroMarginal(mod, 1))
    # Get demand curtailment as required hydropower inputs
    Needed_hydro = EN.get_AllDemandCurtailment(mod)
    print('Required hydro:', Needed_hydro)
    print('Flaf: ', EN.get_HydroFlag(mod, 1))
    # Add hydropower
    EN.set_Hydro(1, Needed_hydro+0.00001+1)
    # Run integrated pyene
    mod = ConcreteModel()
    mod = EN.run(mod)
    print('Marginal ', EN.get_HydroMarginal(mod, 1))
    print('Flaf: ', EN.get_HydroFlag(mod, 1))
    # Get updated demand curtailment
    Demand_curtailed = EN.get_AllDemandCurtailment(mod)
    print('Total curtailment:', Demand_curtailed)

    # 4.25*(7*1 + 2*1) = 29.75
    assert (0.0001 >= abs(Needed_hydro-29.75) and
            0.0001 >= abs(Demand_curtailed))


# Test iteration where hydro covers full demand
def test_pyene_AllHydro():
    '''
    Get all power generation, replace it with hydro, add surplus
    and make sure that it is not used by the pumps
    '''
    print('test_pyene_AllHydro')
    conf = testConfig()
    # Consider two time steps
    conf.NM.settings['NoTime'] = 2  # Number of time steps
    conf.NM.scenarios['Weights'] = [0.5, 1]  # Add weights to the time steps
    # Add hydropower plant
    conf.NM.hydropower['Number'] = 1  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [150]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.01]  # Costs
    # Pumps
    conf.NM.pumps['Number'] = 1  # Number of pumps
    conf.NM.pumps['Bus'] = [2]  # Location (bus) of pumps
    conf.NM.pumps['Max'] = [1000]  # Generation capacity
    conf.NM.pumps['Value'] = [0.001]  # Value/Profit
    # Enable curtailment
    conf.NM.settings['Feasibility'] = True
    # Get Pyene model
    EN = pe(conf.EN)
    # Initialize network model using the selected configuration
    EN.initialise(conf)
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
    # Run model
    mod = ConcreteModel()
    mod = EN.run(mod)
    # Get total energy generation
    Total_Generation = EN.get_AllGeneration(mod)
    print('Total generation: ', Total_Generation)
    # Replace all conventional generation with hydropower
    EN.set_Hydro(1, Total_Generation)
    # Run system again
    mod = ConcreteModel()
    mod = EN.run(mod)
    # Check conventional power generation
    Total_Conv_Generation = EN.get_AllGeneration(mod, 'Conv')
    print('Total conventional generation: ', Total_Conv_Generation)
    # Add even more hydropower
    Additional_hydro = 100
    EN.set_Hydro(1, Total_Generation+Additional_hydro)
    # Run the system again
    mod = ConcreteModel()
    mod = EN.run(mod)
    # Check that the water is spilled instead of used by the pumps
    Hydropower_Left = EN.get_AllHydro(mod)
    print('Hydropower left', Hydropower_Left)

    # 4 .25*(5*(100*0.5+50*1.0)+2*(50+0.5+150+1.0)) = 3612.5
    assert (0.0001 > abs(Total_Generation-3612.5) and
            0.0001 > abs(Total_Conv_Generation) and
            0.0001 > abs(Hydropower_Left-Additional_hydro))


# Test use of renewables and pumps
def test_pyene_RESPump():
    '''
    Set case with surplus RES in one period, and curtailment in another
    Add hydro to cover all conventional genertaion, instead part of it will
    be used to mitigate curtailment. FInally add more hydro to cover both
    conventional generation and curtailment
    '''
    print('test_pyene_RESPump')
    conf = testConfig()
    conf.NM.settings['NoTime'] = 2  # Number of time steps
    conf.NM.scenarios['Weights'] = [0.5, 1]
    # Add hydropower plant
    conf.NM.hydropower['Number'] = 1  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [500]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.01]  # Costs
    # Pumps
    conf.NM.pumps['Number'] = 1  # Number of pumps
    conf.NM.pumps['Bus'] = [2]  # Location (bus) of pumps
    conf.NM.pumps['Max'] = [1000]  # Generation capacity
    conf.NM.pumps['Value'] = [0.001]  # Value/Profit
    # RES generators
    conf.NM.RES['Number'] = 1  # Number of RES generators
    conf.NM.RES['Bus'] = [3]  # Location (bus) of pumps
    conf.NM.RES['Max'] = [100]  # Generation capacity
    conf.NM.RES['Cost'] = [0.0001]  # Costs

    # Enable curtailment
    conf.NM.settings['Feasibility'] = True
    # Get Pyene model
    EN = pe(conf.EN)
    # Initialize network model using the selected configuration
    EN.initialise(conf)
    # Single demand node (first scenario)
    demandNode = _node()
    demandNode.value = [0.2, 0.1]  # DemandProfiles[0][0:conf.Time]
    demandNode.index = 1
    EN.set_Demand(demandNode.index, demandNode.value)
    # Second scenario
    demandNode = _node()
    demandNode.value = [0.1, 1]  # DemandProfiles[1][0:conf.Time]
    demandNode.index = 2
    EN.set_Demand(demandNode.index, demandNode.value)
    # RES profile (first scenario)
    resInNode = _node()
    resInNode.value = [0.5, 1.0]
    resInNode.index = 1
    EN.set_RES(resInNode.index, resInNode.value)
    # RES profile (second scenario)
    resInNode = _node()
    resInNode.value = [0.5, 0.0]
    resInNode.index = 2
    EN.set_RES(resInNode.index, resInNode.value)
    # COnstrain generation
    EN.set_GenCoFlag(1, 200)
    EN.set_GenCoFlag(2, 200)
    mod = ConcreteModel()
    mod = EN.run(mod)
    # Get RES spilled
    RES_Spilled = EN.get_AllRES(mod)
    print('RES spilled ', RES_Spilled)
    # Get use of pumps
    Pumps_Use = EN.get_AllPumps(mod)
    print('Energy used by pumps', Pumps_Use)
    # Get demand curtailed
    Demand_Curtailed = EN.get_AllDemandCurtailment(mod)
    print('Demand curtailed', Demand_Curtailed)
    # Add hydro to replace conventional generation
    Conv_Generation = EN.get_AllGeneration(mod, 'Conv')
    print('Conventional generation ', Conv_Generation)
    EN.set_Hydro(1, Conv_Generation)
    # Run again
    mod = ConcreteModel()
    mod = EN.run(mod)
    # Get new curtailment
    New_Curtailed = EN.get_AllDemandCurtailment(mod)
    print('New demand curtailment ', New_Curtailed)
    # Get use of conventional generation
    Use_ConvGeneration = EN.get_AllGeneration(mod, 'Conv')
    print('Conventional generation ', Use_ConvGeneration)
    # Fully cover conventional generation and demand with hydro
    EN.set_Hydro(1, Conv_Generation+Demand_Curtailed)
    # Run again
    mod = ConcreteModel()
    mod = EN.run(mod)
    # Get use of pumps
    Final_Pump = EN.get_AllPumps(mod)
    print('Energy used by pumps', Final_Pump)
    # Get new curtailment
    Final_Curtailed = EN.get_AllDemandCurtailment(mod)
    print('New demand curtailment ', Final_Curtailed)
    # Get use of conventional generation
    Final_ConvGeneration = EN.get_AllGeneration(mod, 'Conv')
    print('Conventional generation ', Final_ConvGeneration)

    # 4.25*5*50*1 = 1062.5
    # 4.25*2*100*1 = 850
    assert(0.0001 > abs(RES_Spilled) and
           0.0001 > abs(Pumps_Use-1062.5) and
           0.0001 > abs(Demand_Curtailed-850) and
           0.0001 > abs(New_Curtailed) and
           0.0001 > abs(Use_ConvGeneration-Demand_Curtailed) and
           0.0001 > abs(Final_Pump-Pumps_Use) and
           0.0001 > abs(Final_Curtailed) and
           0.0001 > abs(Final_ConvGeneration))


# Test dummy integrated LP
def test_pyene_SingleLP():
    '''
    Assume an external engine couples the pyomo model of pyene with some
    constraints to optimise use of hydropower
    '''
    print('test_pyene_SingleLP')
    conf = testConfig()

    def UpdateConfig(conf):
        conf.EM.settings['File'] = os.path.join(json_directory(),
                                                'TestCase.json')
        conf.NM.settings['NoTime'] = 24  # Number of time steps
        conf.NM.hydropower['Number'] = 3  # Number of hydropower plants
        conf.NM.hydropower['Bus'] = [1, 2, 3]  # Location (bus) of hydro
        conf.NM.hydropower['Max'] = [1000, 1000, 1000]  # Generation capacity
        conf.NM.hydropower['Cost'] = [0.01, 0.01, 0.01]  # Costs
        conf.NM.pumps['Number'] = 2  # Number of pumps
        conf.NM.pumps['Bus'] = [1, 2]  # Location (bus) of pumps
        conf.NM.pumps['Max'] = [1, 1]  # Generation capacity
        conf.NM.pumps['Value'] = [0.001, 0.001]  # Value/Profit
        conf.NM.RES['Number'] = 2  # Number of RES generators
        conf.NM.RES['Bus'] = [1, 2]  # Location (bus) of pumps
        conf.NM.RES['Max'] = [100, 100]  # Generation capacity
        conf.NM.RES['Cost'] = [0, 0]  # Costs
        conf.NM.settings['Feasibility'] = True  # Enable curtailment

        return conf

    def MaxHydroAllowance_rule(m, xv):
        '''Constraint for maximum hydropower allowance'''
        return m.vHydropowerAllowance[xv] <= m.MaxHydro[xv]

    def HydroAllowance_rule(m, xv):
        '''Constraint to link hydropower allowance'''
        return m.vHydropowerAllowance[xv] == m.WInFull[1, xv]-m.WOutFull[1, xv]

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
    '''                         First step
    Assume an external engine creates the pyomo model
    '''
    mod = ConcreteModel()

    '''                         Second step
    The engine creates a pyene object
    '''
    EN = pe(conf.EN)

    '''                          Third step
    pyene is initialised with a predefined configuration (e.g., from MOEA)
    '''
    conf = UpdateConfig(conf)
    EN.initialise(conf)
    # Create conditions for having demand curtailment
    EN.set_GenCoFlag(1, False)  # Switching one generator off
    EN.set_GenCoFlag(2, 499)  # Reducing capacity of the other generator

    '''                        Fourth step
    Initialise pyomo sets, parameters, and variables for pyene
    '''
    mod = EN.build_Mod(EN.EM, EN.NM, mod)

    '''                         Fifth step
    Redefine hydropower inputs as variables
    '''
    del mod.WInFull
    if conf.NM.hydropower['Number'] > 1:
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
    mod.MaxHydro = np.zeros(conf.NM.hydropower['Number'], dtype=float)
    for xh in range(conf.NM.hydropower['Number']):
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
    # Assuming that total hydropower use is assigned to vHydroUse
    mod.WaterValue = 10000
    mod.OF = Objective(rule=OF_rule, sense=minimize)

    '''                        Ninth step
    Running the model
    '''
    # Optimise
    opt = SolverFactory('glpk')
    # Print
    opt.solve(mod)

    '''                            Testing                                  '''
    tstResults = np.zeros(4, dtype=float)
    tstResults[0] = mod.OF.expr()

    mod.WaterValue = 9999
    del mod.OF
    mod.OF = Objective(rule=OF_rule, sense=minimize)
    opt = SolverFactory('glpk')
    opt.solve(mod)
    tstResults[1] = mod.OF.expr()

    mod.WaterValue = 269
    del mod.OF
    mod.OF = Objective(rule=OF_rule, sense=minimize)
    opt = SolverFactory('glpk')
    # Print
    opt.solve(mod)

    tstResults[2] = mod.OF.expr()

    mod.WaterValue = 103
    del mod.OF
    mod.OF = Objective(rule=OF_rule, sense=minimize)
    opt = SolverFactory('glpk')
    # Print
    opt.solve(mod)
    tstResults[3] = mod.OF.expr()

    assert (0.001 >= abs(tstResults[0]-64918496.1787) and
            0.001 >= abs(tstResults[1]-64917775.4642) and
            0.001 >= abs(tstResults[2]-57714991.5907) and
            0.001 >= abs(tstResults[3]-37495351.572))
