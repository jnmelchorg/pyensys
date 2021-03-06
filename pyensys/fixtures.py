import os
from pyensys.engines.main import pyeneConfig


def json_directory():
    ''' Directory contain JSON files for pytest '''
    return os.path.join(os.path.dirname(__file__), '..', 'tests', 'json')


def fixed_config():
    '''Dedicated configuration for pytests '''
    conf = pyeneConfig()
    #TODO: Don't load any file by default
    conf.EM.settings['File'] = os.path.join(json_directory(),
                                            'TreePreload.json')
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
