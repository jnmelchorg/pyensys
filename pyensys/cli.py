import click
import numpy as np
import cProfile
import os
import time
import json
from pyensys.cases import test_pyene, test_pyeneE, test_pyeneN, test_pyeneAC, \
    test_pyenetest, hydro_example_tobeerased, test_pyeneRES
from pyensys.engines.main import pyeneConfig
from pyensys.engines.main import pyeneClass
from pyensys.tests.matpower.conversion_model_mat2json import any2json
from pyensys.Optimisers.input_output_function import  get_peak_data, \
    read_input_data
from pyensys.Optimisers.process_data import mult_for_bus
from pyensys.Optimisers.screening_model_CLI import main_screening
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


@cli.command('run-dist_invest_old')
@click.option('--input_dir', default=
              os.path.join(os.path.dirname(__file__), 'tests', 'json',
                           'attest_input_format_m1.json'),
              help='Loacation and name of input JSON')
@click.option('--output_dir', default=
              os.path.join(os.path.dirname(__file__), 'tests', 'outputs',
                           'output.json'),
              help='Loacation and name of output ')
def pyensys_ATTEST_Distribution_Old(**kwargs):
    ''' Call ATTEST's distribution network planning tool '''
    input_dir = kwargs.pop('input_dir')
    output_dir = kwargs.pop('output_dir')

    start = time.time()
    solution = main_access_function(file_path=input_dir)
    save_in_json(solution, output_dir)
    end = time.time()
    print('\nTime required by the tool:', end - start)


@cli.command('run-dist_invest')
@click.option('--input_dir', default=
              os.path.join(os.path.dirname(__file__), 'tests', 'json',
                           'attest_input_format_m1.json'),
              help='Loacation and name of input JSON')
@click.option('--output_dir', default=
              os.path.join(os.path.dirname(__file__), 'tests', 'outputs',
                           'output.json'),
              help='Loacation and name of output ')
@click.option('--case', default=os.path.join(os.path.dirname(__file__),
                                             'tests', 'matpower', 'case3.m'),
              help='Loacation and name of m file')
@click.option('--line_capacities', default=
              [0.045, 0.075, 0.1125, 0.15, 0.225, 0.3, 0.5, 0.75, 1.0, 2.0,
               5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 250.0,
               500.0], help='Line capacities (list)')
@click.option('--TRS_capacities', default=
              [0.003, 0.006, 0.009, 0.015, 0.03, 0.045, 0.075, 0.1125, 0.15,
               0.225, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0,
               40.0, 50.0, 60.0, 80.0, 100.0, 250.0, 500.0],
              help='Transformer capacities (list)')
@click.option('--line_Costs', default= [], help='Line costs (list)')
@click.option('--TRS_costs', default=[], help='Transformer costs (list)')
@click.option('--cont_list', default=[], help='Contingencies (list)')
@click.option('--growth', default=
              {"Active": {'2020': 0, '2030': 1.89, '2040': 3.0,'2050': 2.5},
               "Slow":{'2020': 0, '2030': 1.1, '2040': 2.0,'2050': 1.0}},
              help='Demand growth (dictionary)')
@click.option('--Max_clusters', default= 5, help='Maximum number of clusters')
def pyensys_ATTEST_Distribution(**kwargs):
    ''' Call ATTEST's distribution network planning tool '''
    Base_Path = os.path.dirname(__file__)
    input_dir = kwargs.pop('input_dir')
    output_dir = kwargs.pop('output_dir')
    test_case = kwargs.pop('case')
    ci_catalogue = [kwargs.pop('line_capacities'),
                    kwargs.pop('trs_capacities')]
    cont_list = kwargs.pop('cont_list')
    NoCon = len(cont_list)

    ci_cost = [[], []]
    if len(ci_cost[0]) == 0:
        ci_cost[0] = [20 * i for i in ci_catalogue[0]]
    else:
        ci_cost[0] = kwargs.pop('line_Costs')

    if len(ci_cost[1]) == 0:
        ci_cost[1] = [20 * i for i in ci_catalogue[1]]
    else:
        ci_cost[0] = kwargs.pop('trs_Costs')

    growth = kwargs.pop('growth')
    Max_clusters = kwargs.pop('max_clusters')

    keys = list(growth.keys())
    yrs = list(growth[keys[0]].keys())
    multiplier = [[1]]
    for yr in range(len(yrs)-1):
        mul = []
        aux = (int(yrs[yr+1])-int(yrs[yr]))/100
        aux1 = multiplier[yr][0]*(1+growth[keys[0]][yrs[yr+1]]*aux)
        aux2 = multiplier[yr][-1]*(1+growth[keys[1]][yrs[yr+1]]*aux)
        aux3 = (aux1-aux2)/(2**(yr+1)-1)
        for bs in range(2**(yr+1)):
            mul.append(aux1)
            aux1 -= aux3
        multiplier.append(mul)

    print('\nScreening for investment options\n')
    gen_status = False
    line_status = True  # Consider lines status in mpc

    file_name = test_case[:-2]
    x = len(file_name)-1
    while file_name[x] != '\\' and file_name[x] != '/':
        x -= 1

    converter = any2json()
    converter.matpower2json(folder_path=file_name[0:x],
                            name_matpower=file_name[x+1:],
                            name_json=file_name[x+1:])
    mpc = json.load(open(os.path.join(file_name[0:x],
                                      file_name[x+1:]+'.json')))
    cont_list = [[1]*mpc['NoBranch']]  # --> do not consider contingencies

    # multipliers for each bus
    busMult_input = []
    # expand multiplier for each bus
    multiplier_bus = mult_for_bus(busMult_input, multiplier, mpc)

    # Load information
    # update peak demand values, get peak load for screening model
    peak_Pd = []  # get_peak_data(mpc, base_time_series_data, peak_hour)

    # Cost information
    # linear cost for the screening model
    cicost = 20  # £/Mw/km --> actually used in the screening model!
    # curtailment cost
    penalty_cost = 1e3

    # Outputs
    interv_dict, interv_clust = \
        main_screening(mpc, gen_status, line_status, multiplier_bus,
                       cicost, penalty_cost, peak_Pd, ci_catalogue,
                       cont_list)

    # reduce catalogue in the interv dictionary
    for xbr in range(mpc["NoBranch"]):
        if sum(interv_dict[xbr]) > 0:
            for xi in range(len(interv_dict[xbr])):
                if mpc["branch"]["TAP"][xbr] == 0:  # line
                    interv_dict[xbr][xi] = \
                        min([i for i in ci_catalogue[0]
                             if i >= interv_dict[xbr][xi]])
                else:  # transformer
                    interv_dict[xbr][xi] = \
                        min([i for i in ci_catalogue[1]
                             if i >= interv_dict[xbr][xi]])

            interv_dict[xbr] = list(set(interv_dict[xbr]))
            interv_dict[xbr].sort()
        else:
            interv_dict[xbr] = []

    final_interv_clust = []

    for i in range(len(interv_clust)):
        fl = False
        for ii in range(len(final_interv_clust)):
            if interv_clust[i] == final_interv_clust[ii]:
                fl = True

        if not fl:
            final_interv_clust.append(interv_clust[i])

    # Limiting number of clusters to use
    NoClusters = len(final_interv_clust)-1
    NoCols = len(final_interv_clust[0])
    if NoClusters > Max_clusters:
        for x1 in range(1, NoClusters+1):
            for x2 in range(NoClusters-x1):
                flg = False
                x3 = 0
                while not flg and x3 < NoCols:
                    if final_interv_clust[x1][x3] > \
                            final_interv_clust[x1+x2+1][x3]:
                        flg = True
                    x3 += 1
                if flg:
                    aux = final_interv_clust[x1]
                    final_interv_clust[x1] = final_interv_clust[x1+x2+1]
                    final_interv_clust[x1+x2+1] = aux
        final_interv_clust = \
            [final_interv_clust[int(plan)]
             for plan in np.ceil(np.linspace(1, NoClusters, Max_clusters))]

    # TODO: Pass data to JSON file

    # Distribution network optimisation
    start = time.time()
    solution = main_access_function(file_path=input_dir)
    save_in_json(solution, output_dir)
    end = time.time()
    print('\nTime required by the tool:', end - start)
