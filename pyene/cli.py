import click
import numpy as np
import cProfile
from .cases import *


class ConfigClass(object):
    def __init__(self):
        self.init = False


pass_conf = click.make_pass_decorator(ConfigClass, ensure=True)


@click.group()
@click.option('--init', is_flag=False, type=bool,
              help='Take the settings from __init__')
@click.option('--hydro', default=3, help='Number of hydropower plants')
@click.option('--profile/--no-porfile', default=False)
@pass_conf
def cli(conf, **kwargs):
    """Prepare pyene simulation"""

    # Initialisation assumptions
    conf.init = kwargs.pop('init')

    # Assume location, capacity and cost of hydro
    conf.NoHydro = kwargs.pop('hydro')
    conf.Hydro = np.zeros(conf.NoHydro, dtype=int)
    conf.HydroMax = np.zeros(conf.NoHydro, dtype=float)
    conf.HydroCost = np.zeros(conf.NoHydro, dtype=float)

    # assume the location of the hydropower plants
    for x in range(conf.NoHydro):
        conf.Hydro[x] = x+1
        conf.HydroMax[x] = 1000
        conf.HydroCost[x] = 0.01

    # Add profiler
    if 'profile' in kwargs:
        profiler = cProfile.Profile()
        profiler.enable()
    else:
        profiler = None


# Update conf based on tree data
def _update_config_pyeneE(conf, kwargs):
    conf.TreeFile = kwargs.pop('tree')

    return conf


# Update config based on network data
def _update_config_pyeneN(conf, kwargs):
    # Number and location of pumps
    conf.NoPump = kwargs.pop('pump')
    conf.Pump = np.zeros(conf.NoPump, dtype=int)
    conf.PumpMax = np.zeros(conf.NoPump, dtype=float)
    conf.PumpVal = np.zeros(conf.NoPump, dtype=float)
    # assume the location of the hydropower plants
    for x in range(conf.NoPump):
        conf.Pump[x] = x+1
        conf.PumpMax[x] = 1
        conf.PumpVal[x] = 0.001

    # Number and location of pumps
    conf.NoRES = kwargs.pop('res')  # Number of RES generators
    conf.NoDemProfiles = 2  # Number of demand profiles
    conf.NoRESProfiles = 2  # Number of RES profiles
    conf.RES = np.zeros(conf.NoRES, dtype=int)
    conf.RESMax = np.zeros(conf.NoRES, dtype=int)
    conf.Cost = np.zeros(conf.NoRES, dtype=float)
    # assume the location of the hydropower plants
    for x in range(conf.NoRES):
        conf.RES[x] = x+1
        conf.Cost[x] = 0
        conf.RESMax[x] = 10

    conf.Security = kwargs.pop('sec')
    conf.Losses = kwargs.pop('loss')
    conf.Feasibility = kwargs.pop('feas')
    conf.NetworkFile = kwargs.pop('network')
    conf.Time = kwargs.pop('time')

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
@click.option('--sec', is_flag=False, type=bool,
              help='Include N-1 security constraints')
@click.option('--loss', is_flag=False, type=bool,
              help='Estimate losses')
@click.option('--feas', is_flag=False, type=bool,
              help='Consider feasibility constratints')
@click.option('--time', default=0, help='Number of time steps')
@pass_conf
def network_simulation_pyeneE(conf, **kwargs):
    """Prepare electricity network simulation"""
    conf=_update_config_pyeneN(conf, kwargs)

    test_pyeneN(conf)


@cli.command('run-en')
@click.option('--tree', default='ResolutionTreeMonth01.json',
              help='Time resolution tree file')
@click.option('--network', default='case14.json',
              help='Network model file')
@click.option('--Pump', default=2, help='Number of pumps')
@click.option('--res', default=2, help='Number of RES generators')
@click.option('--sec', is_flag=False, type=bool,
              help='Include N-1 security constraints')
@click.option('--loss', is_flag=False, type=bool,
              help='Estimate losses')
@click.option('--feas', is_flag=False, type=bool,
              help='Consider feasibility constratints')
@click.option('--time', default=24, help='Number of time steps')
@pass_conf
def network_simulation_pyeneEN(conf, **kwargs):
    """Prepare combined simulation"""
    conf=_update_config_pyeneE(conf, kwargs)
    conf=_update_config_pyeneN(conf, kwargs)
    print(conf)

    test_pyene(conf)
