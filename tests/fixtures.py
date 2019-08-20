import pytest
import os
from pyene.engines.pyene import pyeneConfig


def json_directory():
    """ Directory contain JSON test data. """
    return os.path.join(os.path.dirname(__file__), 'json')


def fixed_config():
    '''Dedicated configuration for the tests '''
    conf = pyeneConfig()
    conf.EM.settings['File'] = os.path.join(json_directory(),
                                            'ResolutionTreeMonth01.json')
    conf.NM.settings['File'] = os.path.join(json_directory(), 'case4.json')

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
    conf.NM.settings['Security'] = []  # List of contingescies to test
    conf.NM.settings['Losses'] = False  # Model losses
    conf.NM.settings['Feasibility'] = False  # Add dummy generators
    conf.NM.scenarios['Weights'] = None  # Weights for each time step
    conf.NM.settings['NoTime'] = 1  # Number of time steps

    # Scenarios
    conf.NM.scenarios['NoDem'] = 2  # Number of demand profiles
    conf.NM.scenarios['NoRES'] = 2  # Number of RES profiles

    # Hydrology
    conf.HM.settings['Flag'] = False  # Disable pyeneH

    return conf

def testConfig():
    return fixed_config()
