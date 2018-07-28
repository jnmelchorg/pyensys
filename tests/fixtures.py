import pytest
import os
from pyene.engines.pyene import pyeneConfig


def json_directory():
    """ Directory contain JSON test data. """
    return os.path.join(os.path.dirname(__file__), 'json')


@pytest.fixture()
def conf():
    # Copy attributes
    obj = pyeneConfig()
    for pars in obj.__dict__.keys():
        setattr(conf, pars, getattr(obj, pars))
    conf.EM.fRea = os.path.join(json_directory(), 'ResolutionTreeMonth01.json')

    conf.NetworkFile = 'case4.json'  # Selected network file
    conf.json = json_directory()  # Location of the json directory

    # Hydropower
    conf.NM.hydropower['Number'] = 0  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = []  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = []  # Generation capacity
    conf.NM.hydropower['Cost'] = []  # Costs

    # Pumps
    conf.NM.pumps['Number'] = 0  # Number of pumps
    conf.NM.pumps['Bus'] = []  # Location (bus) of pumps
    conf.NM.pumps['Max'] = []  # Generation capacity
    conf.NM.pumps['Value'] = []  # Value/Profit

    # RES generators
    conf.NM.RES['Number'] = 0  # Number of RES generators
    conf.NM.RES['Bus'] = []  # Location (bus) of pumps
    conf.NM.RES['Max'] = []  # Generation capacity
    conf.NM.RES['Cost'] = []  # Costs

    # Network considerations
    conf.Security = []  # List of contingescies to test
    conf.Losses = False  # Model losses
    conf.Feasibility = False  # Add dummy generators
    conf.NM.settings['NoTime'] = 0  # Number of time steps

    # Scenarios
    conf.NM.scenarios['NoDem'] = 2  # Number of demand profiles
    conf.NM.scenarios['NoRES'] = 2  # Number of RES profiles
    conf.Weights = None  # Weights for each time step

    return conf
