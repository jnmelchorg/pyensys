import click
import numpy as np
import cProfile
import os
import time
from pyensys.cases import test_pyene, test_pyeneE, test_pyeneN, test_pyeneAC, \
    test_pyenetest, hydro_example_tobeerased, test_pyeneRES, attest_invest, \
    attest_invest_path
from pyensys.engines.main import pyeneConfig
from pyensys.engines.main import pyeneClass
from pyensys.Optimisers.input_output_function import  get_peak_data, \
    read_input_data
from pyensys.managers.GeneralManager import main_access_function, save_in_json


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
    if 'pump' in kwargs.keys():
        NoPump = kwargs.pop('pump')
    else:
        NoPump = 0
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
    if 'res' in kwargs.keys():
        NoRES = kwargs.pop('res')
    else:
        NoRES = 0
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

    if 'sec' in kwargs.keys():
        conf.NM.settings['Security'] = kwargs.pop('sec')  # Contingescies
    else:
        conf.NM.settings['Security'] = []

    if 'loss' in kwargs.keys():
        conf.NM.settings['Losses'] = kwargs.pop('loss')  # Model losses
    else:
        conf.NM.settings['Losses'] = False
    
    if 'feas' in kwargs.keys():
        conf.NM.settings['Feasibility'] = kwargs.pop('feas')  # Dummy generators
    else:
        conf.NM.settings['Feasibility'] = True
    
    conf.NM.scenarios['Weights'] = None  # Weights for each time step
    conf.NM.settings['File'] = os.path.join(os.path.dirname(__file__), 'json',
                                            kwargs.pop('network'))
    # Use linear approximation of losses?
    if 'linearloss' in kwargs.keys():
        aux = kwargs.pop('linearloss')
        if aux > 0:
            conf.NM.settings['Losses'] = False
            conf.NM.settings['Loss'] = aux
    
    # By default pyene will run using glpk
    if 'usepyomo' in kwargs.keys():
        aux = kwargs.pop('usepyomo')
        if aux:
            conf.EN.solverselection['pyomo'] = aux
            conf.EN.solverselection['glpk'] = False        

    # TODO: THIS NEED TO BE REMOVED - IT'S COMPLETELY HARDCODED AND CAN CAUSE 
    # THAT THE CODE CRASHES IN THE FUTURE
    if 'baseline' in kwargs.keys():
        aux = kwargs.pop('baseline')
        conf.NM.hydropower['Baseload'] = [aux, aux]

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
@click.option('--Linearloss', default=0, type=float,
              help='Fraction assigned to losses')
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
              help='Consider feasibility constraints')
@click.option('--time', default=24, help='Number of time steps')
@click.option('--Linearloss', default=0, type=float,
              help='Fraction assigned to losses')
@click.option('--Usepyomo', default=False, type=bool,
              help='Use pyomo for optimisation')
@pass_conf
def network_simulation_pyeneEN(conf, **kwargs):
    """Prepare energy balance and network simulation """
    conf = _update_config_pyeneE(conf, kwargs)
    conf = _update_config_pyeneN(conf, kwargs)

    test_pyene(conf)

@cli.command('run-example-hydro')
@click.option('--tree', default='TreeMonth01_01RD.json',
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
              help='Consider feasibility constraints')
@click.option('--time', default=24, help='Number of time steps')
@click.option('--Linearloss', default=0, type=float,
              help='Fraction assigned to losses')
@click.option('--baseline', default=0, type=float,
              help='Fraction assigned to baseline of hydro electrical generators')
@pass_conf
def network_simulation_pyeneEN(conf, **kwargs):
    """Prepare energy balance and network simulation """
    conf = _update_config_pyeneE(conf, kwargs)
    conf = _update_config_pyeneN(conf, kwargs)

    hydro_example_tobeerased(conf)

@cli.command('run-ac')
@click.option('--tree', default='ResolutionTreeMonth01.json',
              help='Time resolution tree file')
@click.option('--network', default='case14_con.json',
              help='Network model file')
@click.option('--time', default=24, help='Number of time steps')
@pass_conf
def network_simulation_pypsa(conf, **kwargs):
    ''' AC power flow '''
    conf = _update_config_pyeneE(conf, kwargs)
    conf = _update_config_pyeneN(conf, kwargs)

    test_pyeneAC(conf)

@cli.command('run-res')
@click.option('--tree', default='ResolutionTreeMonth01.json',
              help='Time resolution tree file')
@click.option('--Pump', default=0, help='Number of pumps')
@click.option('--sec', default=[], type=list,
              help='Include N-1 security constraints')
@click.option('--network', default='case14_con.json', help='Network model file')
@click.option('--res', default=2,
              help='Number of RES generators')
@click.option('--time', default=24,
              help='Number of time steps')
@click.option('--loss', default=False,
              type=bool, help='Estimate losses')
@click.option('--Linearloss', default=0, type=float,
              help='Fraction assigned to losses')
@click.option('--feas', default=True, type=bool,
              help='Consider feasibility constraints')
@pass_conf
def network_simulation_pyeneRES(conf, **kwargs):
    ''' Prepare Network Simulation '''
    conf = _update_config_pyeneE(conf, kwargs)
    conf = _update_config_pyeneN(conf, kwargs)

    test_pyeneRES(conf)

@cli.command('run-example-hydro')
@click.option('--tree', default='TreeMonth01_01RD.json',
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
              help='Consider feasibility constraints')
@click.option('--time', default=24, help='Number of time steps')
@click.option('--Linearloss', default=0, type=float,
              help='Fraction assigned to losses')
@click.option('--baseline', default=0, type=float,
              help='Fraction assigned to baseline of hydro electrical generators')
@pass_conf
def network_simulation_pyeneHydro(conf, **kwargs):
    """Prepare energy balance and network simulation """
    conf = _update_config_pyeneE(conf, kwargs)
    conf = _update_config_pyeneN(conf, kwargs)

    hydro_example_tobeerased(conf)

@cli.command('run-ac')
@click.option('--tree', default='ResolutionTreeMonth01.json',
              help='Time resolution tree file')
@click.option('--network', default='case14_con.json',
              help='Network model file')
@click.option('--time', default=24, help='Number of time steps')
@pass_conf
def network_simulation_pypsa(conf, **kwargs):
    ''' AC power flow '''
    conf = _update_config_pyeneE(conf, kwargs)
    conf = _update_config_pyeneN(conf, kwargs)

    test_pyeneAC(conf)

@cli.command('test')
@click.option('--test', default=0, help='Example to be executed')
def network_simulation_pyenetst(**kwargs):
    ''' Hidden development functionality '''
    mthd = kwargs.pop('test')
    test_pyenetest(mthd)


@cli.command('run_pyene')
@click.argument('file_path', type=click.Path(exists=True))
def network_simulation_pyenetst(**kwargs):
    ''' run simulation specified in excel file '''
    opt = pyeneClass()
    opt.initialise(path=kwargs.pop('file_path'))
    opt.run()
    opt.save_outputs(sim_no=0)


@cli.command('run')
@click.argument('file_path', type=click.Path(exists=True))
def pyensys_entry_point(**kwargs):
    file_path: str = kwargs.pop('file_path')
    main_access_function(file_path)


@cli.command('run-dist_invest')
@click.option('--output_dir',
              default=os.path.join(os.path.dirname(__file__), 'tests',
                                    'outputs', 'output.json'),
              help='Full path of the outputs. By default:' +
              os.path.join(os.path.dirname(__file__), 'tests', 'outputs',
                            'output.json') + '.')
@click.option('--case', default=os.path.join(os.path.dirname(__file__),
                                              'tests', 'matpower', 'case3.m'),
              help='Location and name of m file. By default' +
              os.path.join(os.path.dirname(__file__), 'tests', 'matpower',
                            'case3.m') + '.')
@click.option('--line_capacities',
              default=[0.045,0.075,0.1125,0.15,0.225,0.3,0.5,0.75,1.0,
                        2.0,3.0,5.0,10.0,15.0,20.0,25.0,30.0,40.0,50.0,60.0,70.0,80.0,
                        100.0,150.0,200.0,250.0,300.0,500.0,750.0,1000,1500,
                        2000,3000,5000,7500,10000,15000,20000,25000,30000,40000,50000],
              help='List of line capacities [MVA]. By default: ' +
              '[0.045, 0.075, 0.1125, 0.15, 0.225, 0.3, 0.5, 0.75, 1.0, ' +
              '2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, ' +
              '250.0, 500.0].')
@click.option('--TRS_capacities',
              default=[0.045,0.075,0.1125,0.15,0.225,0.3,0.5,0.75,1.0,
                        2.0,3.0,5.0,10.0,15.0,20.0,25.0,30.0,40.0,50.0,60.0,70.0,80.0,
                        100.0,150.0,200.0,250.0,300.0,500.0,750.0,1000,1500,
                        2000,3000,5000,7500,10000,15000,20000,25000,30000,40000,50000],
              help='List of transformer capacities [MVA]. By default: ' +
              '[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, ' +
              '100.0, 250.0, 500.0].')
@click.option('--line_costs', default=[],
              help='Line costs [£/km]. By default it is an empty list [] ' +
              'and the model assumes 60000*line capacity. The costs can also ' +
              'be added here, e.g., using the line capacities above: ' +
              '[1800, 3000, 4500, 6000, 9000, 12000, 20000, 30000, 40000, ' +
              '80000, 200000, 400000, 800000, 1200000, 1600000, 2000000, ' +
              '2400000, 3200000, 4000000, 10000000, 20000000].')
@click.option('--TRS_costs', default=[],
              help='Transformer costs [£]. By default it is an empty list []' +
              ' and the model assumes 7000*transformer capacity. The costs ' +
              'can also be added here, e.g., using the transformer ' +
              'capacities above: [7000, 14000, 35000, 70000, 140000, ' +
              '210000, 280000, 350000, 420000, 560000, 700000, 1750000, ' +
              '3500000].')
@click.option('--cont_list', default=[],
              help='List of line contingencies. By default is its empty [].')
@click.option('--line_length',
              default=[], help='Length of each branch [km], by default it' +
              ' is empty [] and the model assumes all lines are 1km long ' +
              'i.e.,[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ' +
              '1, 1, 1].')
@click.option('--growth',
              default={'Active': {'2020': 0, '2030': 1.89, '2040': 3.0, '2050': 3.0},
                        'Slow': {'2020': 0, '2030': 1.1, '2040': 2.0, '2050': 3.0}},
              help='Dictionary with demand growth [%] for selected years ' +
              '(e.g., 2020, 2030 and 2040) and scenarios (e.g., active and ' +
              'slow). The first year is taken as the current year and should' +
              ' be assign a growth of 0%. By default: ' +
              '{\'Active\': {\'2020\': 0, \'2030\': 1.89, \'2040\': 3.0, \'2050\': 3.0},' +
              '\'Slow\': {\'2020\': 0, \'2030\': 1.1, \'2040\': 2.0, \'2050\': 2.0}}.')
@click.option('--add_load_data',
              default=0,
              help='Use additional ATTEST data for EV, PV and storage (EV-PV-Storage_Data_for_Simulations.xlsx). ' +
              'By default: 0 (False). If 1 (True), additional EV data will be added for each bus per each year and scenario.')
@click.option('--add_load_data_case_name',
              default='UK_Dx_01_',
              help='Name of the case for which the addiational load data should be included. ' +
              'This name must be in the Excel sheet format to navigate in the file EV-PV-Storage_Data_for_Simulations.xlsx. ' +
              'By default: UK_Dx_01_ ')
@click.option('--DSR',
              default={'Active': {'2020': 0, '2030': 0.00, '2040': 0.00, '2050': 0.00},
                        'Slow': {'2020': 0, '2030': 0.00, '2040': 0.00, '2050': 0.00}},
              help='Dictionary with DSR [share of flexible demand] for selected years ' +
              '(e.g., 2020, 2030 and 2040) and scenarios (e.g., active and ' +
              'slow). The format of the information must match --growth. ' +
              'By default: {\'Active\': {\'2020\': 0, \'2030\': 0.00, ' +
              '\'2040\': 0.00, \'2050\': 0.00}, \'Slow\': {\'2020\': 0.00, \'2030\': 0.00, ' +
              '\'2040\': 0.00, \'2050\': 0.00}}.' +
              ' [1.00 = 100% of demand for the given year/scenario is flexible; 0.05 = 5% demand flexibility]')
@click.option('--cluster', default=None,
              help='List of investment clusters [MVA]. By default it is ' +
              'set to None and the model will calculate the adequate ' +
              'investment options.')
@click.option('--oversize', default=0,
              help='Option to intentionally oversize investments by ' +
              'selecting the next available value from the ' +
              '--line_capacities. By default it is set to 0 (no oversize).')
@click.option('--Max_clusters', default=3,
              help='Constraint on the maximum number of clusters considered' +
              '. By default it is set to 3.')
@click.option('--scenarios', default=[],
              help='List of scenarios to model. By default it is left empty ' +
              '[] and the model will consider all available scenarios, ' +
              'e.g., [0, 1, 2, 3].')
def pyensys_ATTEST_Distribution(**kwargs):
    ''' Call ATTEST's distribution network planning tool '''
    attest_invest(kwargs)


@cli.command('run-dist_path')
@click.option('--input_dir',
              default=os.path.join(os.path.dirname(__file__), 'tests',
                                   'json', 'attest_input_format_Default.json'),
              help='Loacation and name of output ')
@click.option('--output_dir',
              default=os.path.join(os.path.dirname(__file__), 'tests',
                                   'outputs', 'output.json'),
              help='Loacation and name of output ')
@click.option('--numlines', default=38, help='Number of lines')
def pyensys_ATTEST_Distribution_Path(**kwargs):
    ''' Call ATTEST's distribution network planning tool '''
    attest_invest_path(kwargs)
