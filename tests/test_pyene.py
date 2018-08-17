""" Test the pyeneE engine. """
from fixtures import testConfig, json_directory
from pyene.engines.pyene import pyeneClass as pe
import numpy as np
import os
import json
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
    m = ConcreteModel()
    m = EN.run(m)
    EN.Print_ENSim(m)
    print(m.OF.expr())

    assert 0.0001 >= abs(m.OF.expr()-21952.5*7*4.25)


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
    m = ConcreteModel()
    m = EN.run(m)
    EN.Print_ENSim(m)
    print('\n%f ' % m.OF.expr())

    assert 0.0001 >= abs(m.OF.expr()-527048.8750)


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
    m = ConcreteModel()
    m = EN.run(m)
    print('Marginal ', EN.get_HydroMarginal(m, 1))
    # Get demand curtailment as required hydropower inputs
    Needed_hydro = EN.get_AllDemandCurtailment(m)
    print('Required hydro:', Needed_hydro)
    print('Flaf: ', EN.get_HydroFlag(m, 1))
    # Add hydropower
    EN.set_Hydro(1, Needed_hydro+0.00001+1)
    # Run integrated pyene
    m = ConcreteModel()
    m = EN.run(m)
    print('Marginal ', EN.get_HydroMarginal(m, 1))
    print('Flaf: ', EN.get_HydroFlag(m, 1))
    # Get updated demand curtailment
    Demand_curtailed = EN.get_AllDemandCurtailment(m)
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
    m = ConcreteModel()
    m = EN.run(m)
    # Get total energy generation
    Total_Generation = EN.get_AllGeneration(m)
    print('Total generation: ', Total_Generation)
    # Replace all conventional generation with hydropower
    EN.set_Hydro(1, Total_Generation)
    # Run system again
    m = ConcreteModel()
    m = EN.run(m)
    # Check conventional power generation
    Total_Conv_Generation = EN.get_AllGeneration(m, 'Conv')
    print('Total conventional generation: ', Total_Conv_Generation)
    # Add even more hydropower
    Additional_hydro = 100
    EN.set_Hydro(1, Total_Generation+Additional_hydro)
    # Run the system again
    m = ConcreteModel()
    m = EN.run(m)
    # Check that the water is spilled instead of used by the pumps
    Hydropower_Left = EN.get_AllHydro(m)
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
    m = ConcreteModel()
    m = EN.run(m)
    EN.Print_ENSim(m)
    # Get RES spilled
    RES_Spilled = EN.get_AllRES(m)
    print('RES spilled ', RES_Spilled)
    # Get use of pumps
    Pumps_Use = EN.get_AllPumps(m)
    print('Energy used by pumps', Pumps_Use)
    # Get demand curtailed
    Demand_Curtailed = EN.get_AllDemandCurtailment(m)
    print('Demand curtailed', Demand_Curtailed)
    # Add hydro to replace conventional generation
    Conv_Generation = EN.get_AllGeneration(m, 'Conv')
    print('Conventional generation ', Conv_Generation)
    EN.set_Hydro(1, Conv_Generation)
    # Run again
    m = ConcreteModel()
    m = EN.run(m)
    # Get new curtailment
    New_Curtailed = EN.get_AllDemandCurtailment(m)
    print('New demand curtailment ', New_Curtailed)
    # Get use of conventional generation
    Use_ConvGeneration = EN.get_AllGeneration(m, 'Conv')
    print('Conventional generation ', Use_ConvGeneration)
    # Fully cover conventional generation and demand with hydro
    EN.set_Hydro(1, Conv_Generation+Demand_Curtailed)
    # Run again
    m = ConcreteModel()
    m = EN.run(m)
    # Get use of pumps
    Final_Pump = EN.get_AllPumps(m)
    print('Energy used by pumps', Final_Pump)
    # Get new curtailment
    Final_Curtailed = EN.get_AllDemandCurtailment(m)
    print('New demand curtailment ', Final_Curtailed)
    # Get use of conventional generation
    Final_ConvGeneration = EN.get_AllGeneration(m, 'Conv')
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
        return m.vHydropowerAllowance[xv] == m.vEIn[1, xv]-m.vEOut[1, xv]

    def ZeroHydroIn_rule(m, xn, xv):
        '''Constraint to link hydropower allowance'''
        return m.vEIn[xn, xv] == 0

    def HyroActualUse_rule(m, xv):
        '''Constraint for actual hydropower used'''
        return m.vHydroUse[xv] == m.vEIn[1, xv]-m.vEOut[1, xv]

    def AdjustPyeneMod(m, EN):
        '''Final modifications for pyene to work with the integrated model'''
        # Set unused water inputs to zero
        m.sN0 = [x for x in range(EN.EM.LL['NosBal']+1)]
        m.sN0.pop(1)
        m.ZeroHydroIn = Constraint(m.sN0, EN.EM.s['Vec'],
                                   rule=ZeroHydroIn_rule)

        # Variables used by pyene for claculating the objective function
        m = EN._AddPyeneCons(m)
        EN.OFaux = EN._Calculate_OFaux()
        m.OFh = EN.NM.connections['set']
        m.OFhGC = EN.NM.connections['Cost']
        m.OFFea = EN.NM.connections['Feasibility']
        m.OFpenalty = EN.Penalty
        m.OFpumps = EN.NM.pumps['Value']
        m.base = EN.NM.networkE.graph['baseMVA']
        m.OFhDL = EN.NM.connections['Pump']
        m.OFweights = EN.NM.scenarios['Weights']
        m.OFaux = EN.OFaux

        return m

    def OF_rule(m):
        ''' Combined objective function '''
        return (m.WaterValue*sum(m.vEIn[1, xv] for xv in m.sVec) +
                sum((sum(sum(m.vNGCost[m.OFhGC[xh]+xg, xt] for xg in m.sNGen) +
                         sum(m.vNFea[m.OFFea[xh]+xf, xt] for xf in m.sNFea) *
                         m.OFpenalty for xt in m.sNTim) -
                     sum(m.OFpumps[xdl]*m.base *
                         sum(m.vNPump[m.OFhDL[xh]+xdl+1, xt] *
                             m.OFweights[xt]
                             for xt in m.sNTim) for xdl in m.sNDL)) *
                    m.OFaux[xh] for xh in m.OFh))
    '''                         First step
    Assume an external engine creates the pyomo model
    '''
    m = ConcreteModel()

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
    m = EN.build_Mod(m)

    '''                         Fifth step
    Redefine hydropower inputs as variables
    '''
    del m.vEIn
    if conf.NM.hydropower['Number'] > 1:
        m.vEIn = Var(EN.EM.s['Nodz'], EN.EM.s['Vec'],
                     domain=NonNegativeReals, initialize=0.0)
    else:
        m.vEIn = Var(EN.EM.s['Nodz'], domain=NonNegativeReals,
                     initialize=0.0)

    '''                         Sixth step
    Define hydropower allowance
    Assuming pywr were to have these values in a variable called
    vHydropowerAllowance
    '''
    m.vHydropowerAllowance = Var(EN.EM.s['Vec'], domain=NonNegativeReals,
                                 initialize=0.0)
    m.MaxHydro = np.zeros(conf.NM.hydropower['Number'], dtype=float)
    for xh in range(conf.NM.hydropower['Number']):
        m.MaxHydro[xh] = 134000

    # Add constraint to limit hydropower allowance
    m.MaxHydroAllowance = Constraint(EN.EM.s['Vec'],
                                     rule=MaxHydroAllowance_rule)

    # Link hydropower constraint to pyene
    m.HydroAllowance = Constraint(EN.EM.s['Vec'], rule=HydroAllowance_rule)

    '''                        Seventh step
    Make final modifications to pyene, so that the new constraints and
    objective function work correctly
    '''
    m = AdjustPyeneMod(m, EN)

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
    m.WaterValue = 10000
    m.sVec = EN.EM.s['Vec']
    m.sNGen = EN.NM.s['Gen']
    m.sNFea = EN.NM.s['Fea']
    m.sNTim = EN.NM.s['Tim']
    m.sNDL = EN.NM.s['Pump']
    m.OF = Objective(rule=OF_rule, sense=minimize)

    '''                        Ninth step
    Running the model
    '''
    # Optimise
    opt = SolverFactory('glpk')
    # Print
    opt.solve(m)

    '''                            Testing                                  '''
    tstResults = np.zeros(4, dtype=float)
    tstResults[0] = m.OF.expr()

    m.WaterValue = 9999
    del m.OF
    m.OF = Objective(rule=OF_rule, sense=minimize)
    opt = SolverFactory('glpk')
    opt.solve(m)
    tstResults[1] = m.OF.expr()

    m.WaterValue = 269
    del m.OF
    m.OF = Objective(rule=OF_rule, sense=minimize)
    opt = SolverFactory('glpk')
    # Print
    opt.solve(m)

    tstResults[2] = m.OF.expr()

    m.WaterValue = 103
    del m.OF
    m.OF = Objective(rule=OF_rule, sense=minimize)
    opt = SolverFactory('glpk')
    # Print
    opt.solve(m)
    tstResults[3] = m.OF.expr()

    assert (0.001 >= abs(tstResults[0]-64918496.1787) and
            0.001 >= abs(tstResults[1]-64917775.4642) and
            0.001 >= abs(tstResults[2]-57714991.5907) and
            0.001 >= abs(tstResults[3]-37495351.572))


# Combined use of pyeneE, pyeneN and pyeneH
def test_pyene_ENH():
    print('test_pyene_ENH')
    conf = testConfig()
    # Hydropower
    conf.NM.hydropower['Number'] = 3  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [2, 3, 4]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [1000, 1000, 1000]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.01, 0.01,  0.01]  # Costs
    conf.HM.hydropower['Node'] = [6, 3, 5]  # Location (node)  of hydro
    conf.HM.hydropower['Efficiency'] = [0.85, 0.85, 0.85]  # pu
    conf.HM.hydropower['Head'] = [200, 200, 200]  # m

    # Study settings
    conf.NM.settings['NoTime'] = 24  # Number of time steps
    conf.NM.scenarios['NoDem'] = 2  # Number of demand profiles
    conf.NM.settings['Pieces'] = [10]  # 10 MW pieces

    # Enable pyeneH
    conf.HM.settings['Flag'] = True
    conf.HM.rivers['DepthMin'] = [0.3, 0.3, 0.3, 0.3]  # MInimum depth

    # Create object
    EN = pe(conf.EN)

    # Initialise with selected configuration
    EN.initialise(conf)

    # Profiles and water allowance
    fileName = os.path.join(json_directory(), 'UKElectricityProfiles.json')
    Eprofiles = json.load(open(fileName))
    EN.set_Demand(1, Eprofiles['Winter']['Weekday'])
    EN.set_Demand(2, Eprofiles['Winter']['Weekend'])
    EN.set_Hydro(1, 12000)
    EN.set_Hydro(2, 35000)

    # Run integrated pyene
    m = ConcreteModel()
    m = EN.run(m)
    EN.Print_ENSim(m)
    print(m.OF.expr())

    assert 0.0001 >= abs(m.OF.expr()-5993235.52384) and \
        0.0001 >= abs(m.vHin[1, 17].value-102.1684) and \
        0.0001 >= abs(m.vHin[3, 18].value-58.9768)


# Combined use of pyeneE, pyeneN and pyeneH + Storage
def test_pyene_ENHStor():
    print('test_pyene_ENHStor')
    conf = testConfig()
    # Hydropower
    conf.NM.hydropower['Number'] = 3  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [2, 3, 4]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [1000, 1000, 1000]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.01, 0.01,  0.01]  # Costs
    conf.HM.hydropower['Node'] = [6, 3, 5]  # Location (node)  of hydro
    conf.HM.hydropower['Efficiency'] = [0.85, 0.85, 0.85]  # pu
    conf.HM.hydropower['Head'] = [200, 200, 200]  # (m)
    conf.HM.hydropower['Storage'] = [0, 100, 50]  # local storage

    # Study settings
    conf.NM.settings['NoTime'] = 24  # Number of time steps
    conf.NM.scenarios['NoDem'] = 2  # Number of demand profiles
    conf.NM.settings['Pieces'] = [10]  # 10 MW pieces

    # Enable pyeneH
    conf.HM.settings['Flag'] = True
    conf.HM.rivers['DepthMin'] = [0.3, 0.3, 0.3, 0.3]  # MInimum depth

    # Create object
    EN = pe(conf.EN)

    # Initialise with selected configuration
    EN.initialise(conf)

    # Profiles and water allowance
    fileName = os.path.join(json_directory(), 'UKElectricityProfiles.json')
    Eprofiles = json.load(open(fileName))
    EN.set_Demand(1, Eprofiles['Winter']['Weekday'])
    EN.set_Demand(2, Eprofiles['Winter']['Weekend'])
    EN.set_Hydro(1, 12000)
    EN.set_Hydro(2, 35000)

    # Run integrated pyene
    m = ConcreteModel()
    m = EN.run(m)
    EN.Print_ENSim(m)
    print(m.OF.expr())

    assert 0.0001 >= abs(m.OF.expr()-5946534.84017) and \
        0.0001 >= abs(m.vHin[1, 17].value-113.9128) and \
        0.0001 >= abs(m.vHin[3, 17].value-51.7954)


# Combined use of pyeneE, pyeneN and pyeneH + Storage
def test_pyene_ENHStorPump():
    print('test_pyene_ENHStorPump')
    conf = testConfig()
    # Scenarios
    conf.NM.scenarios['NoDem'] = 2  # Number of demand profiles
    conf.NM.scenarios['NoRES'] = 2  # Number of RES profiles

    # Hydropower
    conf.NM.hydropower['Number'] = 3  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [2, 3, 4]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [1000, 1000, 1000]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.01, 0.01,  0.01]  # Costs
    conf.HM.hydropower['Node'] = [6, 3, 5]  # Location (node)  of hydro
    conf.HM.hydropower['Efficiency'] = [0.85, 0.85, 0.85]  # pu
    conf.HM.hydropower['Head'] = [200, 200, 200]  # m
    conf.HM.hydropower['Storage'] = [0, 100, 50]  # local storage

    # Pumps
    conf.NM.pumps['Number'] = 3  # Number of pumps
    conf.NM.pumps['Bus'] = [1, 2, 3]  # Location (bus) of pumps
    conf.NM.pumps['Max'] = [100, 100, 100]  # Capacity
    conf.NM.pumps['Value'] = [0.1, 0.1, 0.1]  # Value/Profit
    conf.HM.pumps['From'] = [2, 6, 3]  # Location in water network
    conf.HM.pumps['To'] = [5, 0, 4]  # Location in (or out) the water network
    conf.HM.pumps['Efficiency'] = [0.85, 0.85, 0.85]  # pu
    conf.HM.pumps['Head'] = [200, 200, 200]  # m

    # RES generators
    conf.NM.RES['Number'] = 2  # Number of RES generators
    conf.NM.RES['Bus'] = [1, 3]  # Location (bus) of pumps
    conf.NM.RES['Max'] = [100, 100]  # Generation capacity
    conf.NM.RES['Cost'] = [0.0001, 0.0001]  # Costs

    # Enable curtailment
    conf.NM.settings['Feasibility'] = False
    conf.NM.settings['NoTime'] = 24  # Number of time steps

    conf.HM.rivers = {
            'DepthMax': [4, 4, 4, 4, 4],  # Maximum depth
            'DepthMin': [0.3, 0.3, 0.3, 0.3, 0.3],  # MInimum depth
            'From': [1, 2, 4, 4, 3],  # Node - from
            'Length': [1000, 1000, 1000, 1000, 1000],  # length (m)
            'Manning': [0.03, 0.03, 0.03, 0.03, 0.03],  # Mannings 'n
            'Parts': [],
            'Share': [1, 1, 0.4, 0.6, 1],  # Links between water flows
            'Slope': [0.0001, 0.0001, 0.0001, 0.0001, 0.0001],  # Slope (m)
            'To': [2, 3, 5, 6, 7],  # Node -to
            'Width': [200, 200, 200, 200, 200]  # width (m)
            }
    conf.HM.nodes['Out'] = [7, 5, 6]  # Nodes with water outflows
    conf.HM.settings['Flag'] = True
    conf.NM.settings['Pieces'] = [10]

    # Create object
    EN = pe(conf.EN)

    # Initialise with selected configuration
    EN.initialise(conf)

    # Profiles and water allowance
    fileName = os.path.join(json_directory(), 'UKElectricityProfiles.json')
    Eprofiles = json.load(open(fileName))
    EN.set_Demand(1, Eprofiles['Winter']['Weekday'])
    EN.set_Demand(2, Eprofiles['Winter']['Weekend'])
    EN.set_Hydro(1, 12000)
    EN.set_Hydro(2, 35000)

    # RES profile (first scenario)
    aux = np.ones(conf.NM.settings['NoTime'], dtype=int)
    for xr in range(conf.NM.RES['Number']):
        EN.set_RES(xr+1, aux)

    # Run integrated pyene
    m = ConcreteModel()
    m = EN.run(m)
    EN.Print_ENSim(m)
    print(m.OF.expr())
    print(m.vHin[1, 8].value)
    print(m.vHin[4, 9].value)
    print(m.vHStor[1, 6].value)
    print(m.vHStor[2, 12].value)
    print(m.vNPump[2, 5].value*100)
    print(m.vNPump[6, 5].value*100)

    assert 0.0001 >= abs(m.OF.expr()-1306400.90327) and \
        0.0001 >= abs(m.vHin[1, 8].value-44.4162) and \
        0.0001 >= abs(m.vHin[4, 9].value-22.4070) and \
        0.0001 >= abs(m.vHStor[1, 6].value-29.0000) and \
        0.0001 >= abs(m.vHStor[2, 12].value-20.0795) and \
        0.0001 >= abs(m.vNPump[2, 5].value*100-11.7158) and \
        0.0001 >= abs(m.vNPump[6, 5].value*100-17.2354)
