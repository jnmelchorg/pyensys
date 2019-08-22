import click
import numpy as np
import cProfile
import os
from .cases import test_pyene, test_pyeneE, test_pyeneN, test_pyenetest
from .engines.pyene import pyeneConfig


pass_conf = click.make_pass_decorator(pyeneConfig, ensure=True)


@click.group()
@click.option('--init', is_flag=False, type=bool,
              help='Take the settings from __init__')
@click.option('--hydro', default=3, help='Number of hydropower plants')
@click.option('--profile/--no-profile', default=False)
@pass_conf
def cli(conf, **kwargs):
    """Prepare pyene simulation"""

    # Assume location, capacity and cost of hydro
    NoHydro = kwargs.pop('hydro')
    conf.NM.hydropower['Number'] = NoHydro
    conf.NM.hydropower['Bus'] = np.zeros(NoHydro, dtype=int)
    conf.NM.hydropower['Max'] = np.zeros(NoHydro, dtype=float)
    conf.NM.hydropower['Cost'] = np.zeros(NoHydro, dtype=float)

    # assume the location of the hydropower plants
    for x in range(NoHydro):
        conf.NM.hydropower['Bus'][x] = x+1
        conf.NM.hydropower['Max'][x] = 1000
        conf.NM.hydropower['Cost'][x] = 0.01

    # Add profiler
    if 'profile' in kwargs:
        profiler = cProfile.Profile()
        profiler.enable()
    else:
        profiler = None


# Update conf based on tree data
def _update_config_pyeneE(conf, kwargs):
    conf.EM.settings['File'] = os.path.join(os.path.dirname(__file__), 'json',
                                            kwargs.pop('tree'))

    return conf


# Update config based on network data
def _update_config_pyeneN(conf, kwargs):
    # Number and location of pumps
    NoPump = kwargs.pop('pump')
    conf.NM.pumps['Number'] = NoPump
    conf.NM.pumps['Bus'] = np.zeros(NoPump, dtype=int)
    conf.NM.pumps['Max'] = np.zeros(NoPump, dtype=float)
    conf.NM.pumps['Value'] = np.zeros(NoPump, dtype=float)
    # assume the location of the hydropower plants
    for x in range(NoPump):
        conf.NM.pumps['Bus'][x] = x+1
        conf.NM.pumps['Max'][x] = 1
        conf.NM.pumps['Value'][x] = 0.001

    # Number and location of RES
    NoRES = kwargs.pop('res')
    conf.NM.RES['Number'] = NoRES
    conf.NM.RES['Bus'] = np.zeros(NoRES, dtype=int)
    conf.NM.RES['Max'] = np.zeros(NoRES, dtype=float)
    conf.NM.RES['Cost'] = np.zeros(NoRES, dtype=float)
    # assume the location of the hydropower plants
    for x in range(NoRES):
        conf.NM.RES['Bus'][x] = x+1
        conf.NM.RES['Max'] = 0
        conf.NM.RES['Cost'] = 10

    conf.NM.scenarios['NoDem'] = 2  # Number of demand profiles
    conf.NM.scenarios['NoRES'] = 2  # Number of RES profiles
    conf.NM.settings['NoTime'] = kwargs.pop('time')  # Time steps per scenario

    conf.NM.settings['Security'] = kwargs.pop('sec')  # Contingescies to test
    conf.NM.settings['Losses'] = kwargs.pop('loss')  # Model losses
    conf.NM.settings['Feasibility'] = kwargs.pop('feas')  # Dummy generators
    conf.NM.scenarios['Weights'] = None  # Weights for each time step
    conf.NM.settings['File'] = os.path.join(os.path.dirname(__file__), 'json',
                                            kwargs.pop('network'))

    return conf


@cli.command('run-e')
@click.option('--tree', default='ResolutionTreeYear03.json',
              help='Time resolution tree file')
@pass_conf
def energy_balance_pyeneE(conf, **kwargs):
    """Prepare energy balance simulation"""
    conf = _update_config_pyeneE(conf, kwargs)

    test_pyeneE(conf)


@cli.command('run-n')
@click.option('--network', default='case4.json',
              help='Network model file')
@click.option('--pump', default=0, help='Number of pumps')
@click.option('--res', default=0, help='Number of RES generators')
@click.option('--sec', default=[], type=list,
              help='Include N-1 security constraints')
@click.option('--loss', default=False, type=bool,
              help='Estimate losses')
@click.option('--feas', default=False, type=bool,
              help='Consider feasibility constratints')
@click.option('--time', default=1, help='Number of time steps')
@pass_conf
def network_simulation_pyeneE(conf, **kwargs):
    """Prepare electricity network simulation"""
    conf = _update_config_pyeneN(conf, kwargs)
    print('\n\nChecking results\n\n')
    print(conf.NM.settings)
    print('\n\nDt\n\n')

    test_pyeneN(conf)


@cli.command('run-en')
@click.option('--tree', default='ResolutionTreeMonth01.json',
              help='Time resolution tree file')
@click.option('--network', default='case14_con.json',
              help='Network model file')
@click.option('--Pump', default=0, help='Number of pumps')
@click.option('--res', default=0, help='Number of RES generators')
@click.option('--sec', default=[], type=list,
              help='Include N-1 security constraints')
@click.option('--loss', default=True, type=bool,
              help='Estimate losses')
@click.option('--feas', default=True, type=bool,
              help='Consider feasibility constratints')
@click.option('--time', default=24, help='Number of time steps')
@pass_conf
def network_simulation_pyeneEN(conf, **kwargs):
    """Prepare combined simulation"""
    conf = _update_config_pyeneE(conf, kwargs)
    conf = _update_config_pyeneN(conf, kwargs)

    test_pyene(conf)


@cli.command('test')
@click.option('--test', default=0, help='Example to be executed')
def network_simulation_pyenetst(**kwargs):
    ''' Hidden development functionality '''
    mthd = kwargs.pop('test')
    test_pyenetest(mthd)
