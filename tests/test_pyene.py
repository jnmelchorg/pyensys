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
    print('test_pyene_Small')
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
    hydroInNode = _node()
    for xh in range(conf.NoHydro):
        hydroInNode.value = 1000
        hydroInNode.index = xh+1
        EN.set_Hydro(hydroInNode)
    # Run integrated pyene
    mod = EN.run()
    EN.Print_ENSim(mod, EN.EM, EN.NM)
    print('\n%f ' % mod.OF.expr())

    assert 0.0001 >= abs(mod.OF.expr()-527048.8750)


# Converting to pypsa
def test_pyene2pypsa(conf):
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
    print('\n\nCalling pyene\n\n')
    (nu, pypsaFlag) = EN.pyene2pypsa(xscen)    
    # Run pypsa
    if pypsaFlag:
        nu.pf()
        aux1 = nu.lines_t.p0['Line1'][0]
        aux2 = 158.093958
    else:
        aux1 = 0
        aux2 = 0
    print(pypsaFlag)
    assert 0.0001 >= abs(aux1 - aux2)
