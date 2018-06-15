""" Test the pyeneE engine. """
from click.testing import CliRunner
from fixtures import *
from pyene.engines.pyene import pyeneClass as pe
import numpy as np
import os

# Interaction node
class _node():
    def __init__(self):
        self.value = None
        self.index = None
        self.bus = None
        self.marginal = None
        self.flag = False


# Energy balance and network simulation
def test_pyene_Small(conf):
    print('test_pyene_Small')
    conf.NetworkFile = 'case4.json'
    conf.TreeFile = 'ResolutionTreeMonth01.json'
    conf.Time = 1  # Single period
    # Create object
    EN = pe()
    # Initialise with selected configuration
    EN.initialise(conf)
    # Run integrated pyene
    mod = EN.run()
    EN.Print_ENSim(mod, EN.EM, EN.NM)
    print(mod.OF.expr())

    assert 0.0001 >= abs(mod.OF.expr()-21952.5*7*4.25)


# Adding hydropower
def test_pyene_SmallHydro(conf):
    print('test_pyene_SmallHydro')
    conf.NetworkFile = 'case4.json'
    conf.TreeFile = 'ResolutionTreeMonth01.json'
    conf.Time = 1  # Single period

    # Adding hydropower plants
    conf.NoHydro = 2
    conf.Hydro = np.zeros(conf.NoHydro, dtype=int)
    conf.HydroMax = np.zeros(conf.NoHydro, dtype=float)
    conf.HydroCost = np.zeros(conf.NoHydro, dtype=float)
    for x in range(conf.NoHydro):
        conf.Hydro[x] = x+1
        conf.HydroMax[x] = 100
        conf.HydroCost[x] = 0.01

    # Create object
    EN = pe()
    # Initialise with selected configuration
    EN.initialise(conf)
    # Add hydro nodes
    for xh in range(conf.NoHydro):
        EN.set_Hydro(xh+1, 1000)
    # Run integrated pyene
    mod = EN.run()
    EN.Print_ENSim(mod, EN.EM, EN.NM)
    print('\n%f ' % mod.OF.expr())

    assert 0.0001 >= abs(mod.OF.expr()-527048.8750)


# Converting to pypsa
def test_pyene2pypsa(conf):
    print('test_pyene2pypsa')
    # Selected network file
    conf.NetworkFile = 'case14.json'
    # Location of the json directory
    conf.json = conf.json = os.path.join(os.path.dirname(__file__), 'json')
    # Define number of time spets
    conf.Time = 1  # Number of time steps
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
    (nu, pypsaFlag) = EN.pyene2pypsa(xscen)
    # Run pypsa
    nu.pf()

    assert 0.0001 >= abs(nu.lines_t.p0['Line1'][0] - 158.093958)


# Test iteration where hydro covers load curtailment
def test_pyene_Curtailment2Hydro(conf):
    '''
    Identify demand curtailment in a first iteration
    and supply customers with hydropower in another.
    '''
    print('test_pyene_Curtailment2Hydro')
    # Selected network file
    conf.NetworkFile = 'case4.json'
    # Location of the json directory
    conf.json = conf.json = os.path.join(os.path.dirname(__file__), 'json')
    # Consider single time step
    conf.Time = 1  # Number of time steps
    # Add hydropower plant
    conf.NoHydro = 1  # Number of hydropower plants
    conf.Hydro = [1]  # Location (bus) of hydro
    conf.HydroMax = [100]  # Generation capacity
    conf.HydroCost = [0.01]  # Costs
    # Enable curtailment
    conf.Feasibility = True
    # Get Pyene model
    EN = pe()
    # Initialize network model using the selected configuration
    EN.initialise(conf)
    # one generator off
    EN.set_GenCoFlag(1, False)
    # reduce capcity of the other generator
    EN.set_GenCoFlag(2, 499)
    # Run integrated pyene
    mod = EN.run()
    # Get demand curtailment as required hydropower inputs
    Needed_hydro = EN.get_AllDemandCurtailment(mod)
    print('Required hydro:', Needed_hydro)
    # Add hydropower
    EN.set_Hydro(1, Needed_hydro+0.00001)
    # Run integrated pyene
    mod = EN.run()
    # Get updated demand curtailment
    Demand_curtailed = EN.get_AllDemandCurtailment(mod)
    print('Total curtailment:', Demand_curtailed)

    # 4.25*(7*1 + 2*1) = 29.75
    assert (0.0001 >= abs(Needed_hydro-29.75) and
            0.0001 >= abs(Demand_curtailed))


# Test iteration where hydro covers full demand
def test_pyene_AllHydro(conf):
    '''
    Get all power generation, replace it with hydro, add surplus
    and make sure that it is not used by the pumps
    '''
    print('test_pyene_AllHydro')
    # Selected network file
    conf.NetworkFile = 'case4.json'
    # Location of the json directory
    conf.json = conf.json = os.path.join(os.path.dirname(__file__), 'json')
    # Consider two time steps
    conf.Time = 2  # Number of time steps
    conf.Weights = [0.5, 1]  # Add different weights to the time steps
    # Add hydropower plant
    conf.NoHydro = 1  # Number of hydropower plants
    conf.Hydro = [1]  # Location (bus) of hydro
    conf.HydroMax = [150]  # Generation capacity
    conf.HydroCost = [0.01]  # Costs

    # Pumps
    conf.NoPump = 1  # Number of pumps
    conf.Pump = [2]  # Location (bus) of pumps
    conf.PumpMax = [1000]  # Generation capacity
    conf.PumpVal = [0.001]  # Value/Profit
    # Enable curtailment
    conf.Feasibility = True
    # Get Pyene model
    EN = pe()
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

    # 4 .25*(5*(100*0.5+50*1.0)+2*(50+0.5+150+1.0)) = 3612.5
    assert (0.0001 > abs(Total_Generation-3612.5) and
            0.0001 > abs(Total_Conv_Generation) and
            0.0001 > abs(Hydropower_Left-Additional_hydro))


# Test use of renewables and pumps
def test_pyene_RESPump(conf):
    '''
    Set case with surplus RES in one period, and curtailment in another
    Add hydro to cover all conventional genertaion, instead part of it will
    be used to mitigate curtailment. FInally add more hydro to cover both
    conventional generation and curtailment
    '''
    print('test_pyene_RESPump')
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
    # RES profile (first scenario)
    resInNode = _node()
    resInNode.value = [0.5, 0.0]
    resInNode.index = 2
    EN.set_RES(resInNode.index, resInNode.value)
    # COnstrain generation
    EN.set_GenCoFlag(1, 200)
    EN.set_GenCoFlag(2, 200)
    mod = EN.run()
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
    mod = EN.run()
    # Get new curtailment
    New_Curtailed = EN.get_AllDemandCurtailment(mod)
    print('New demand curtailment ', New_Curtailed)
    # Get use of conventional generation
    Use_ConvGeneration = EN.get_AllGeneration(mod, 'Conv')
    print('Conventional generation ', Use_ConvGeneration)
    # Fully cover conventional generation and demand with hydro
    EN.set_Hydro(1, Conv_Generation+Demand_Curtailed)
    # Run again
    mod = EN.run()
    # Get use of pumps
    Final_Pump = EN.get_AllPumps(mod)
    print('Energy used by pumps', Final_Pump)
    # Get new curtailment
    Final_Curtailed = EN.get_AllDemandCurtailment(mod)
    print('New demand curtailment ', Final_Curtailed)
    # Get use of conventional generation
    Final_ConvGeneration = EN.get_AllGeneration(mod, 'Conv')
    print('Conventional generation ', Final_ConvGeneration)

    #4.25*5*50*1 = 1062.5
    #4.25*2*100*1 = 850
    assert(0.0001 > abs(RES_Spilled) and
           0.0001 > abs(Pumps_Use-1062.5) and
           0.0001 > abs(Demand_Curtailed-850) and
           0.0001 > abs(New_Curtailed) and
           0.0001 > abs(Use_ConvGeneration-Demand_Curtailed) and
           0.0001 > abs(Final_Pump-Pumps_Use) and
           0.0001 > abs(Final_Curtailed) and
           0.0001 > abs(Final_ConvGeneration))
