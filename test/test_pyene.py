""" Test the pyeneE engine. """
from click.testing import CliRunner
from fixtures import *
from pyene.engines.pyene import pyeneClass as pe
import numpy as np

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
        EN.loadHydro(hydroInNode)
    # Run integrated pyene
    mod = EN.run()
    EN.Print_ENSim(mod, EN.EM, EN.NM)
    print('\n%f ' % mod.OF.expr())

    assert 0.0001 >= abs(mod.OF.expr()-527048.8750)
