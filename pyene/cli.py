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


# Update config based on tree data
def _update_config_pyeneE(config, kwargs):
    config.TreeFile = kwargs.pop('tree')

    return config


# Update config based on network data
def _update_config_pyeneN(config, kwargs):
    #Number and location of pumps
    config.NoPump = kwargs.pop('pump')
    config.Pump = np.zeros(config.NoPump)
    # assume the location of the hydropower plants
    for x in range(config.NoPump):
        config.Pump[x] = x+1

    # Number and location of pumps
    config.NoRES = kwargs.pop('res')
    config.RES = np.zeros(config.NoRES)
    # assume the location of the hydropower plants
    for x in range(config.NoRES):
        config.RES[x] = x+1

    config.Security = kwargs.pop('sec')
    config.Losses = kwargs.pop('loss')
    config.Feasibility = kwargs.pop('feas')
    config.NetworkFile = kwargs.pop('network')

    return config


@cli.command('run-e')
@click.option('--tree', default='ResolutionTreeYear03.json',
              help='Time resolution tree file')
@pass_config
def energy_balance_pyeneE(config, **kwargs):
    """Prepare energy balance simulation"""
    config=_update_config_pyeneE(config, kwargs)

    test_pyeneE(config)


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
@pass_config
def network_simulation_pyeneE(config, **kwargs):
    """Prepare electricity network simulation"""
    config=_update_config_pyeneN(config, kwargs)

    test_pyeneN(config)


@cli.command('run-en')
@click.option('--tree', default='ResolutionTreeYear03.json',
              help='Time resolution tree file')
@click.option('--network', default='case14.json',
              help='Network model file')
@click.option('--Pump', default=1, help='Number of pumps')
@click.option('--res', default=0, help='Number of RES generators')
@click.option('--sec', is_flag=False, type=bool,
              help='Include N-1 security constraints')
@click.option('--loss', is_flag=False, type=bool,
              help='Estimate losses')
@click.option('--feas', is_flag=False, type=bool,
              help='Consider feasibility constratints')
@pass_config
def network_simulation_pyeneEN(config, **kwargs):
    """Prepare electricity network simulation"""
    config=_update_config_pyeneE(config, kwargs)
    config=_update_config_pyeneN(config, kwargs)

    test_pyene(config)
