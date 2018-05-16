import click
import numpy as np
from .cases import *


class Config(object):
    def __init__(self):
        self.init = False


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option('--init', is_flag=False, type=bool,
              help='Take the settings from __init__')
@click.option('--hydro', default=1, help='Number of hydropower plants')
@pass_config
def cli(config, **kwargs):
    """Prepare pyene simulation"""

    # Initialisation assumptions
    config.init = kwargs.pop('init')

    #Number and location of hydro
    config.NoHydro = kwargs.pop('hydro')
    config.Hydro = np.zeros(config.NoHydro)
    # assume the location of the hydropower plants
    for x in range(config.NoHydro):
        config.Hydro[x] = x+1


@cli.command('run-e')
@click.option('--tree', default='ResolutionTreeYear03.json',
              help='Time resolution tree file')
@pass_config
def energy_balance_pyeneE(config, **kwargs):
    """Prepare energy balance simulation"""
    config.TreeFile = kwargs.pop('tree')
    test_pyeneE(config)


@cli.command('run-n')
@click.option('--network', default='case4.json',
              help='Network model file')
@click.option('--Pump', default=1, help='Number of pumps')
@pass_config
def network_simulation_pyeneE(config, **kwargs):
    """Prepare electricity network simulation"""

    #Number and location of pumps
    config.NoPump = kwargs.pop('pump')
    config.Pump = np.zeros(config.NoHydro)
    # assume the location of the hydropower plants
    for x in range(config.NoHydro):
        config.Hydro[x] = x+1

    config.NetworkFile = kwargs.pop('network')
    test_pyeneN(config)
