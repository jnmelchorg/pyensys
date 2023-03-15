""" This module contains the FutureDAMS - pyene test case definitions.

These test cases are used to assess the different functionalities of the
engine.

# TODO complete this description once the cases are written.

"""
from .engines.main import pyeneClass, pyeneConfig
from pyomo.core import ConcreteModel
from pyomo.environ import SolverFactory
from .engines.pyeneO import PrintinScreen as PSc
import os
import json
import time
import numpy as np
from pyensys.tests.matpower.conversion_model_mat2json import any2json
from pyensys.Optimisers.process_data import mult_for_bus
from pyensys.Optimisers.screening_model_CLI import main_screening
from pyensys.managers.GeneralManager import main_access_function, save_in_json


def get_pyene(conf=None):
    """ Get pyene object."""

    return pyeneClass(conf)


def get_pyeneConfig():
    """ Get pyene object."""

    return pyeneConfig()


def test_pyeneAC(config):
    """ Run pypsa based AC power flow """
    config.NM.hydropower['Number'] = 0  # Number of hydropower plants
    
    # Create object
    EN = pyeneClass(config.EN)

    # Initialise with selected configuration
    EN.initialise(config)

    # Fake weather engine
    FileName = 'TimeSeries.json'
    (DemandProfiles, NoDemPeriod, BusDem, LinkDem, NoRES, NoRESP,
     LLRESType, LLRESPeriod, RESProfs, RESBus, RESLink, NoLink,
     Nohr) = EN.ReadTimeS(FileName)
    
    # Profiles
    demandNode = _node()
    demandNode.value = DemandProfiles[0][:]
    demandNode.index = 1
    EN.set_Demand(demandNode.index, demandNode.value)

    # Second scenario
    demandNode = _node()
    demandNode.value = DemandProfiles[1][:]
    demandNode.index = 2
    EN.set_Demand(demandNode.index, demandNode.value)
    
    EN_Interface = EN.getClassInterfaces()

    Model_Pypsa = [EN_Interface.pyene2pypsa(EN.NM, x)[0]
                   for x in range(config.NM.scenarios['NoDem'])]

    for xscen in range(config.NM.scenarios['NoDem']):
        Model_Pypsa[xscen].pf()

    EN.NM.print(Model_Pypsa, range(config.NM.scenarios['NoDem']), None, True)

def test_pyeneE(config):
    """ Execute pyene to access pyeneE - Full json based simulation."""
    EN = pyeneClass(config.EN)
    (EM, EModel, results) = EN.ESim(config)
    EM.print(EModel)


# Network simulation test
def test_pyeneN(config):
    """ Execute pyene to access pyeneN - Full json based simulation."""
    # Create object
    EN = pyeneClass(config.EN)
    # Run model
    (NM, NModel, results) = EN.NSim(config)
    print('\n\nOF: ', NModel.OF.expr())
    NM.print(NModel)
    


# Interaction node
class _node():
    def __init__(self):
        self.value = None
        self.index = None
        self.bus = None
        self.marginal = None
        self.flag = False


# pyene simulation test
def test_pyene(conf):
    """ Execute pyene to access pyeneN - Full json based simulation."""
    # Disable pyeneH
    conf.HM.settings['Flag'] = False

    # Create object
    EN = pyeneClass(conf.EN)

    # Initialise with selected configuration
    EN.initialise(conf)

    # Fake weather engine
    FileName = 'TimeSeries.json'
    (DemandProfiles, NoDemPeriod, BusDem, LinkDem, NoRES, NoRESP,
     LLRESType, LLRESPeriod, RESProfs, RESBus, RESLink, NoLink,
     Nohr) = EN.ReadTimeS(FileName)

    # Single demand node (first scenario)
    demandNode = _node()
    demandNode.value = DemandProfiles[0][:]
    demandNode.index = 1
    EN.set_Demand(demandNode.index, demandNode.value)

    # Second scenario
    demandNode = _node()
    demandNode.value = DemandProfiles[1][:]
    demandNode.index = 2
    EN.set_Demand(demandNode.index, demandNode.value)

    # Several RES nodes
    resInNode = _node()
    for xr in range(EN.NM.RES['Number']):
        resInNode.value = RESProfs[xr][:]
        resInNode.index = xr+1
        EN.set_RES(resInNode.index, resInNode.value)

    # Several hydro nodes
    hydroInNode = _node()
    for xh in range(EN.NM.hydropower['Number']):
        hydroInNode.value = 0
        hydroInNode.index = xh+1
        EN.set_Hydro(hydroInNode.index, hydroInNode.value)

    if conf.EN.solverselection['pyomo']:
        # Run integrated pyene
        mod = ConcreteModel()
        mod = EN.run(mod)

        # Print results
        print('\n\nOF: ', mod.OF.expr())
        # EN.NM.offPrint()
        # EN.NM.Print['Generation'] = True
        # EN.NM.Print['Losses'] = True
        # EN.NM.Print['Flows'] = True
        # EN.NM.Print['GenBus'] = True
        EN.Print_ENSim(mod)

        # Collect unused hydro:
        print()
        hydroOutNode = _node()
        for xh in range(EN.EM.size['Vectors']):
            hydroOutNode.index = xh+1
            print('Hydro %d', hydroOutNode.index, ' left: ',
                  EN.get_Hydro(mod, hydroOutNode.index), ' (',
                  EN.get_HydroMarginal(mod, hydroOutNode.index), ')',
                  EN.get_HydroFlag(mod, hydroOutNode.index))

        # Collect output of pumps
        print()
        pumpNode = _node()
        for xp in range(EN.NM.pumps['Number']):
            pumpNode.index = xp+1
            pumpNode.value = EN.get_Pump(mod, xp+1)
            print('Pump %d: %f' % (pumpNode.index, pumpNode.value))

        # Collect RES spill
        print()
        resOutNode = _node()
        for xp in range(EN.NM.RES['Number']):
            resOutNode.index = xp+1
            resOutNode.value = EN.get_RES(mod, resOutNode)
            print('RES %d: %f' % (resOutNode.index, resOutNode.value))

        # # Collect curtailment per node
        # print()
        # curNode = _node()
        # for xn in range(EN.NM.ENetwork.get_NoBus()):
        #     curNode.bus = xn+1
        #     curNode.value = EN.get_DemandCurtailment(mod, curNode.bus)
        #     print('Dem %d: %f' % (curNode.bus, curNode.value))

        # Collect all curtailment
        print()
        curAll = _node()
        curAll.value = EN.get_AllDemandCurtailment(mod)
        print('Total curtailment:', curAll.value)

    if conf.EN.solverselection['glpk']:
        # Energy model in glpk
        from .engines.pyene_Models import EnergyandNetwork as ENMod
        Model = ENMod(EN.EM, EN.NM, EN)
        Model.optimisationENM()
        # Print results
        print('\n\nOF: ', Model.GetObjectiveFunctionENM())
        Prints = PSc(EN)
        Prints.PrintallResults(Model)

        # Collect unused hydro:
        print()
        OutputsTree = Model.GetOutputsTree()
        EnergybalanceDual = Model.GetEnergybalanceDual()
        aux1 = Model.PenaltyCurtailment/Model.BaseUnitPower
        hydroOutNode = _node()
        for xv in range(Model.NumberTrees):
            hydroOutNode.index = xv+1
            if -EnergybalanceDual[xv, 1] > aux1:
                aux2 = True
            else:
                aux2 = False
            print('Hydro %d', hydroOutNode.index, ' left: ',
                  OutputsTree[xv, 1], ' (',
                  int(EnergybalanceDual[xv, 1]), ')',
                  aux2)

        # Collect output of pumps
        print()
        if Model.NumberPumps > 0:
            pumpNode = _node()
            PumpOperation = Model.GetPumpOperation()
            for xp in range(Model.NumberPumps):
                value = 0
                for xh in Model.LongTemporalConnections:
                    for xt in range(Model.ShortTemporalConnections):
                        value += PumpOperation[xh, xt, xp]
                pumpNode.index = xp+1
                pumpNode.value = value
                print('Pump %d: %f' % (pumpNode.index, pumpNode.value))

        # Collect RES spill
        print()
        if Model.NumberRESGen > 0:
            resOutNode = _node()
            RESGeneration = Model.GetRESGeneration()
            for xp in range(Model.NumberRESGen):
                value = 0
                for xh in Model.LongTemporalConnections:
                    acu = 0
                    for xt in range(Model.ShortTemporalConnections):
                        value += (Model.MaxRESGen[xp]*\
                            Model.RESScenarios[xh, xt, xp] \
                            * Model.BaseUnitPower -
                            RESGeneration[xh, xt, xp])
                resOutNode.index = xp+1
                resOutNode.value = value
                print('RES %d: %f' % (resOutNode.index, resOutNode.value))

        # # Collect curtailment per node
        # print()
        # curNode = _node()
        # for xn in range(EN.NM.ENetwork.get_NoBus()):
        #     curNode.bus = xn+1
        #     curNode.value = EN.get_DemandCurtailment(mod, curNode.bus)
        #     print('Dem %d: %f' % (curNode.bus, curNode.value))

        # Collect all curtailment
        print()
        curAll = _node()
        values = [0, 0]
        value = 0
        if Model.FlagProblem and Model.FlagFeasibility:
            LoadCurtailment = Model.GetLoadCurtailmentNodes()
        else:
            LoadCurtailment = Model.GetLoadCurtailmentSystemED()
        for xs in Model.LongTemporalConnections:
            for xt in range(Model.ShortTemporalConnections):
                if Model.FlagProblem and LoadCurtailment is not None:
                    for k in range(Model.NumberContingencies + 1):
                        for ii in range(Model.NumberNodesPS):
                            value += LoadCurtailment[xs, xt, k, ii]
                            values[EN.NM.ENetwork.Bus[ii].get_LT()] += \
                                LoadCurtailment[xs, xt, k, ii]
                if not Model.FlagProblem and LoadCurtailment is not None:
                    value += LoadCurtailment[xs, xt]
        curAll.value = value, values
        print('Total curtailment:', curAll.value)

def test_pyeneRES(conf):
    '''
    A case study with 2 PV generators located at two buses for 2 representative winter days
    '''
    # TODO THIS NEEDS TO BE ERASED IN A FUTURE RELEASE
    # Initialise simulation methods
    conf.HM.settings['Flag'] = False
    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True
    conf.NM.settings['NoGenerators'] = 12
    #conf.NM.settings['File'] = os.path.join(json_directory(), 'caseGhana_Sim40_BSec_ManualV02.json')
    conf.NM.settings['NoTime'] = 24
    conf.NM.scenarios['Weights'] = [1 for _ in range(conf.NM.settings['NoTime'])]
    #conf.NM.scenarios['Number'] = 4  # Four scenarios including winter weekday/weekend and summer weekday/weekend
    conf.NM.scenarios['NoDem'] = 2  # Four demand profiles (repeated once) including winter weekday/weekend and summer weekday/weekend

    # RES generators
    conf.NM.RES['Number'] = 2  # Number of RES generators
    conf.NM.RES['Bus'] = [1, 4]  # Location (bus) of generators
    conf.NM.RES['Max'] = [10, 15]  # Generation capacity
    conf.NM.RES['Cost'] = [0.001, 0.001]  # Costs

    # Hydro generators
    conf.NM.hydropower['Number'] = 1  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [5]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [200]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.001]  # Costs

    # Create object
    EN = pe(conf.EN)

    # Initialise with selected configuration
    EN.initialise(conf)

    ################### case 1 - Winter scenarios ###########################

    # # Demand profile
    # fileName = os.path.join(json_directory(), 'TimeSeries.json')
    # Eprofiles = json.load(open(fileName))
    # EN.set_Demand(1, Eprofiles['Demand']['Values'][0])  # First scenario demand profile - Winter Weekday
    # if conf.NM.scenarios['NoDem'] == 2:
    #     EN.set_Demand(2, Eprofiles['Demand']['Values'][1])  # Second scenario demand profile - Winter Weekend
    #
    # # RES profile
    # resInNode = _node()
    # for xr in range(EN.NM.RES['Number']):
    #     resInNode.value = Eprofiles['PV']['Values'][xr]  # Winter profile for RES generator 1 & 2
    #     resInNode.index = xr + 1
    #     EN.set_RES(resInNode.index, resInNode.value)
    #
    # # Hydro profile
    # hydroInNode = _node()
    # for xh in range(EN.NM.hydropower['Number']):
    #     hydroInNode.value = 1000  # Winter dry seasons MWh
    #     hydroInNode.index = xh + 1
    #     EN.set_Hydro(hydroInNode.index, hydroInNode.value)
    #
    # # Solve the model and generate results
    # m = ConcreteModel()
    # m = EN.run(m)
    # EN.Print_ENSim(m)
    # print('Total curtailment:', EN.get_AllDemandCurtailment(m))
    # print('Spill ', EN.get_AllRES(m))
    # print('OF   : ', m.OF.expr())

    ################### case 2 - Summer scenarios ###########################

    # Demand profile
    fileName = os.path.join(json_directory(), 'TimeSeries.json')
    Eprofiles = json.load(open(fileName))
    EN.set_Demand(1, Eprofiles['Demand']['Values'][4])  # Third scenario demand profile - Summer Weekday
    if conf.NM.scenarios['NoDem'] == 2:
        EN.set_Demand(2, Eprofiles['Demand']['Values'][5])  # Fourth scenario demand profile - Summer Weekend

    # RES profile
    resInNode = _node()
    for xr in range(EN.NM.RES['Number']):
        resInNode.value = Eprofiles['PV']['Values'][xr+4]  # Summer profile for RES generator 1 & 2
        resInNode.index = xr + 1
        EN.set_RES(resInNode.index, resInNode.value)

    # Hydro profile
    hydroInNode = _node()
    for xh in range(EN.NM.hydropower['Number']):
        hydroInNode.value = 10000  # Summer rainy seasons MWh
        hydroInNode.index = xh + 1
        EN.set_Hydro(hydroInNode.index, hydroInNode.value)

    # Solve the model and generate results
    m = ConcreteModel()
    m = EN.run(m)
    EN.Print_ENSim(m)
    print('Total curtailment:', EN.get_AllDemandCurtailment(m))
    print('Spill ', EN.get_AllRES(m))
    print('OF   : ', m.OF.expr())

def hydro_example_tobeerased(conf):
    """ Execute pyene to run the example of baseload"""
    # TODO THIS NEEDS TO BE ERASED IN A FUTURE RELEASE
    # Disable pyeneH
    conf.HM.settings['Flag'] = False
    conf.NM.settings['NoGenerators'] = 13

def hydro_example_tobeerased(conf):
    """ Execute pyene to run the example of baseload - THIS NEEDS TO BE ERASED \
        IN A FUTURE RELEASE"""
    # Disable pyeneH
    conf.HM.settings['Flag'] = False

    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True
    conf.NM.settings['NoTime'] = 24  # Single period
    conf.NM.scenarios['Weights'] = [1 for _ in range(24)]  # Add weights 
        # to the time steps

    # Adding hydropower plants
    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 5]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [200, 200]  # Generation capacity
    conf.NM.hydropower['Cost'] = [0.001, 0.001]  # Costs

    # Create object
    EN = pyeneClass(conf.EN)

    # Initialise with selected configuration
    EN.initialise(conf)

    # Single demand node (first scenario)
    demandNode = _node()
    demandNode.value = [0.3333, 0.2623, 0.2350, 0.2240, 0.2240, 
        0.2514, 0.3825, 0.6284, 0.6503, 0.5574, 0.5301, 0.5137, 
        0.5355, 0.5027, 0.4918, 0.5464, 0.7760, 0.9891, 1.0000, 
        0.9399, 0.8634, 0.8142, 0.6885, 0.4918]
    demandNode.index = 1
    EN.set_Demand(demandNode.index, demandNode.value)

    # Several hydro nodes
    hydroInNode = _node()
    for xh in range(EN.NM.hydropower['Number']):
        hydroInNode.value = 10000
        hydroInNode.index = xh+1
        EN.set_Hydro(hydroInNode.index, hydroInNode.value)

    # Run integrated pyene
    m = ConcreteModel()
    m = EN.run(m)
    EN.Print_ENSim(m)
    print('Total curtailment:', EN.get_AllDemandCurtailment(m))
    print('Spill ', EN.get_AllRES(m))
    print('OF   : ', m.OF.expr())


def test_pyenetest(mthd):
    '''Test specific functionalities'''
    from .engines.pyeneT import TestClass

    txt = 'test' + str(mthd)
    method_to_call = getattr(TestClass, txt)
    method_to_call(TestClass)


def get_mpc(test_case):
    '''Load MPC data'''
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

    return mpc


def Sceenning_clusters(gen_status, line_status, test_case, multiplier,
                       flex, ci_catalogue, cont_list, Max_clusters):
    '''Produce list of investment clusters'''
    mpc = get_mpc(test_case)
    cont_list = [[1]*mpc['NoBranch']]  # --> do not consider contingencies

    PD_sum = sum(mpc["bus"]["PD"])
    QD_sum = sum(mpc["bus"]["QD"])

    Q_load_correction = 1 + QD_sum/PD_sum # increase line capacity to match Q demand as well (DCOPF screening model cannot include Q)

    # multipliers for each bus
    busMult_input = []
    # expand multiplier for each bus
    multiplier_bus = mult_for_bus(busMult_input, multiplier, flex, mpc)

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
                       cont_list, Q_load_correction)


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
    
    print("interv_clust: ",interv_clust)
    # sum_interv_clust = 0
    # for i in range(len(interv_clust)):
    #     sum_interv_clust += sum(interv_clust[i])
    # if sum_interv_clust == 0:
    #     print()
    #     print("NO INVESTMENTS IDENTIFIED BY THE SCREENING MODEL: THE PLANNING MODEL ABORTS HERE")
    #     print()
    for i in range(len(interv_clust)):
        fl = False
        for ii in range(len(final_interv_clust)):
            if interv_clust[i] == final_interv_clust[ii]:
                fl = True
                print("interv_clust[i] == final_interv_clust[ii] ...")

        if not fl:
            final_interv_clust.append(interv_clust[i])
            print("appending final_interv_clust...")

    # Limiting number of clusters to use
    NoClusters = len(final_interv_clust)-1
    NoCols = len(final_interv_clust[0])
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
    print("NoClusters = ",NoClusters)
    print("Max_clusters = ",NoClusters)
    if NoClusters > Max_clusters:
        final_interv_clust = \
            [final_interv_clust[int(plan)]
             for plan in np.ceil(np.linspace(1, NoClusters, Max_clusters))]
        print("NoClusters > Max_clusters")
    else:
        print("NoClusters < Max_clusters")
        pos = []
        for x1 in range(NoClusters+1):
            flg = True
            x2 = 0
            while flg and x2 < NoCols:
                if final_interv_clust[x1][x2] > 0:
                    flg = False
                x2 += 1
            if not flg:
                pos.append(x1)
        final_interv_clust = [final_interv_clust[x] for x in pos]

    print("Sceenning_clusters... final_interv_clust = ",final_interv_clust)
    return final_interv_clust, mpc


def build_json(test_case, multiplier, mpc, final_interv_clust, yrs,
               ci_cost, ci_catalogue, line_length, scenarios, oversize):
    '''Create main json input file'''
    # Build structure of json file
    data = {
        'problem': {
            'inter-temporal_opf': False,
            'return_rate_in_percentage': 3.0,
            'non_anticipative': True
            },
        'pandapower_mpc_settings': {
            'mat_file_path': test_case,
            'frequency': 60.0
            },
        'optimisation_profiles_data': {
            'format_data': 'attest',
            'data': [{
                'group': 'buses',
                'data': [],
                'columns_names': ['scenario', 'year', 'bus_index', 'p_mw',
                                  'q_mvar']
                }]
            },
        'pandapower_optimisation_settings': {
            'display_progress_bar': True,
            'optimisation_software': 'pypower'
            },
        'optimisation_binary_variables': [{
            'element_type': "line",
            'costs': [],
            'elements_positions': [],
            'installation_time': [],
            'capacity_to_be_added_MW': []
            }]
        }

    # Add demand growth scenarios

    # find position of buses with non-zero demand
    pos_demand = []
    PD = mpc['bus']['PD']
    QD = mpc['bus']['QD']
    for xd in range(len(PD)):
        if PD[xd] != 0:
            pos_demand.append(xd)

    # Build scenarios
    NoYrs = len(yrs)
    NoSce = len(multiplier[NoYrs-1])
    if len(scenarios) == 0:
        scenarios = range(NoSce)
    else:
        scenarios = [int(x) for x in scenarios]
    for xs in scenarios:
        for xy in range(NoYrs):
            pos = int(np.ceil((xs+1)/2**(NoYrs-1)*2**xy)-1)
            for xb in pos_demand:
                (data['optimisation_profiles_data']['data'][0]['data'].
                 append([xs+1, int(yrs[xy]), xb, PD[xb]*multiplier[xy][pos],
                         QD[xb]*multiplier[xy][pos]]))

    # Add investment clusters
    NoClu = len(final_interv_clust)
    print("final_interv_clust = ",final_interv_clust)
    NoBra = len(final_interv_clust[0])
    for xc in range(NoClu):
        costs = 0
        elements = []
        Ins_time = 0
        capacity = []
        for xb in range(NoBra):
            if final_interv_clust[xc][xb] > 0:
                x = 0
                while final_interv_clust[xc][xb] != ci_catalogue[0][x]:
                    x += 1
                x += oversize  # Oversize
                costs += ci_cost[0][x]*line_length[xb]
                elements.append(xb+1)
                capacity.append(ci_catalogue[0][x])
        data['optimisation_binary_variables'][0]['costs'].append(costs)
        (data['optimisation_binary_variables'][0]
        ['elements_positions'].append(elements))
        (data['optimisation_binary_variables'][0]
        ['installation_time'].append(Ins_time))
        (data['optimisation_binary_variables'][0]
        ['capacity_to_be_added_MW'].append(capacity))

    return data


def kwargs2ListF(kwargs, txt):
    '''Convert string to floating list'''
    var = kwargs.pop(txt)
    if isinstance(var, str):
        res = []
        for s in var:
            if s.isdigit():
                res.append(float(s))
    else:
        res = var

    return res


def save_in_jsonW(solution, output_dir, NoLines, clusters_positions,
                  clusters_capacity, case_name, yrs=[], scen=[]):
    '''Save results in json file'''
    data = {
        'Country': case_name,
        'Case name': case_name,
        'Scenario 1': {
            'Total investment cost (EUR-million)': 0,
            'Flexibility investment cost (EUR-million)': 0,
            'Net Present Operation Cost (EUR-million)': 0
            },
        'Scenario 2': {
            'Total investment cost (EUR-million)': 0,
            'Flexibility investment cost (EUR-million)': 0,
            'Net Present Operation Cost (EUR-million)': 0
            }
        }

    # Get first and last scenarios and the position of the data
    NoRes = len(solution[0]['data']['scenario'])  # Number of results
    scenarios = [1000, 0]
    pos = [[], []]
    for xs in range(NoRes):
        scen = solution[0]['data']['scenario'][xs]
        if scenarios[0] > scen:
            pos[0] = []
            scenarios[0] = scen

        if scenarios[1] < scen:
            pos[1] = []
            scenarios[1] = scen

        if scenarios[0] == scen:
            pos[0].append(xs)

        if scenarios[1] == scen:
            pos[1].append(xs)

    # Only one scenario was available
    if scenarios[0] == scenarios[1]:
        pos[1] = []

    # Get years
    if len(yrs) == 0:
        NoYrs = 0
        yrs = []
        for xp in range(2):
            for xy in pos[xp]:
                y = solution[0]['data']['year'][xy]
                flg = True
                x = 0
                while flg and x < NoYrs:
                    if yrs[x] == y:
                        flg = False
                    x += 1
                if flg:
                    NoYrs += 1
                    yrs.append(y)
    else:
        NoYrs = len(yrs)

    for xp in range(2):
        for xy in yrs:
            data['Scenario ' + str(xp+1)][str(xy)] = {
                'Operation cost (EUR-million/year)': 0,
                'Branch investment (MVA)': [0 for x in range(NoLines)],
                'Flexibility investment (MW)': [0 for x in range(NoLines)]
                }

    # Add interventions
    for xp in range(2):
        txt = 'Scenario ' + str(xp+1)
        for xy in pos[xp]:
            yr = str(solution[0]['data']['year'][xy])
            lin_cluster = solution[0]['data']['line_index'][xy]
            NoC1 = len(lin_cluster)
            # Find corresponding cluster
            xc1 = -1
            flg = True
            while flg:
                xc1 += 1
                NoC2 = len(clusters_positions[xc1])
                if NoC1 == NoC2:
                    xc2 = 0
                    while xc2 < NoC1 and flg:
                        if lin_cluster[xc2] != clusters_positions[xc1][xc2]:
                            flg = False
                        xc2 += 1
                    if NoC1 == xc2:
                        flg = False
            for xc in range(NoC1):
                ax = clusters_positions[xc1][xc]
                data[txt][yr]['Branch investment (MVA)'][ax] = \
                    clusters_capacity[xc1][xc]

    for xs in solution[1]['data'].values:
        if xs[0] == 0:
            data['Scenario 1']['Total investment cost (EUR-million)'] = \
                xs[1]/1000000
        if xs[0] == scen:
            data['Scenario 2']['Total investment cost (EUR-million)'] = \
                xs[1]/1000000

    # Save output file
    with open(output_dir, 'w') as fp:
        json.dump(data, fp, indent=4)


def attest_invest(kwargs):
    '''Call ATTEST's distribution network planning tool '''
    Base_Path = os.path.dirname(__file__)
    output_dir = kwargs.pop('output_dir')
    test_case = kwargs.pop('case')
    ci_catalogue = [kwargs2ListF(kwargs, 'line_capacities'),
                    kwargs2ListF(kwargs, 'trs_capacities')]
    cont_list = kwargs.pop('cont_list')
    cluster = kwargs.pop('cluster')
    line_length = kwargs2ListF(kwargs, 'line_length')
    scenarios = kwargs2ListF(kwargs, 'scenarios')
    oversize = kwargs.pop('oversize')
    # NoCon = len(cont_list)

    ci_cost = [[], []]
    line_costs = kwargs.pop('line_costs')
    if isinstance(line_costs, str):
        line_costs = eval(line_costs)
    if len(line_costs) == 0:
        ci_cost[0] = [40000 * i for i in ci_catalogue[0]]
    else:
        ci_cost[0] = line_costs

    trs_costs = kwargs.pop('trs_costs')
    if isinstance(trs_costs, str):
        trs_costs = eval(trs_costs)
    if len(trs_costs) == 0:
        ci_cost[1] = [7000 * i for i in ci_catalogue[1]]
    else:
        ci_cost[0] = trs_costs

    growth = kwargs.pop('growth')
    if isinstance(growth, str):
        growth = eval(growth)

    DSR = kwargs.pop('dsr')
    Max_clusters = kwargs.pop('max_clusters')

    keys = list(growth.keys())
    yrs = list(growth[keys[0]].keys())
    multiplier = [[1*(1+growth[keys[0]][yrs[0]])]]

    flex = [[1-DSR[keys[0]][yrs[0]]]]
    for yr in range(len(yrs)-1):
        mul = []
        aux = (int(yrs[yr+1])-int(yrs[yr]))/100
        aux1 = multiplier[yr][0]*(1+growth[keys[0]][yrs[yr+1]]*aux)
        aux2 = multiplier[yr][-1]*(1+growth[keys[1]][yrs[yr+1]]*aux)
        aux3 = (aux1-aux2)/(2**(yr+1)-1)

        dsr = []
        dsrax1 = 1-DSR[keys[0]][yrs[yr+1]]
        dsrax2 = (dsrax1+DSR[keys[1]][yrs[yr+1]]-1)/(2**(yr+1)-1)

        for bs in range(2**(yr+1)):
            mul.append(aux1)
            dsr.append(dsrax1)
            aux1 -= aux3
            dsrax1 -= dsrax2
        flex.append(dsr)
        multiplier.append(mul)
    
    print('\nmultiplier (growth input) = ')
    print(multiplier)
    print()

    print('\nScreening for investment options\n')
    NoScens = len(multiplier[-1])-1
    if cluster is None:
        gen_status = False
        line_status = True
        final_interv_clust, mpc = \
            Sceenning_clusters(gen_status, line_status, test_case, multiplier,
                               flex, ci_catalogue, cont_list, Max_clusters)
    else:
        final_interv_clust = eval(cluster)
        mpc = get_mpc(test_case)

    # Check line lengths
    NoLines = len(mpc['branch']['F_BUS'])
    if len(line_length) == 0:
        line_length = [1 for x in range(NoLines)]
    else:
        for x in range(NoLines-len(line_length)):
            line_length.append(line_length[0])

    if final_interv_clust == []:
        print()
        print("NO INVESTMENTS IDENTIFIED BY THE SCREENING MODEL: THE PLANNING MODEL ABORTS HERE")
        print()
        # Save output file
        with open(output_dir, 'w') as fp:
            json.dump("NO INVESTMENTS IDENTIFIED BY THE SCREENING MODEL: THE PLANNING MODEL WAS ABORTED", fp, indent=4)
    else:
        # # manually add interventions for tests:
        # final_interv_clust = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]]
        
        # # Pass data to JSON file
        filename = os.path.join(Base_Path, 'tests', 'json', 'ATTEST_Inputs.json')
        with open(filename, 'w', encoding='utf-8') as f:
            data = build_json(test_case, multiplier, mpc, final_interv_clust, yrs,
                            ci_cost, ci_catalogue, line_length, scenarios,
                            oversize)
            json.dump(data, f, ensure_ascii=False, indent=4)

        # Distribution network optimisation
        print('\nOptimising investment strategies\n')
        start = time.time()
        solution, info = main_access_function(file_path=filename)
        
        if info.incumbent_interventions._container == []:
            print()
            print("WARNING: THE INVESTMENT PLANNING MODEL FAILED TO FIND INTERVENTIONS - PLEASE CHECK FEASIBILITY AND CONVERGENCE OF ACOPF MODELS")
            print()
            # Save output file
            with open(output_dir, 'w') as fp:
                json.dump("THE INVESTMENT PLANNING MODEL FAILED TO FIND INTERVENTIONS - PLEASE CHECK FEASIBILITY AND CONVERGENCE OF ACOPF MODELS", fp, indent=4)
        else:
            # # Get clusters
            clusters_positions = []
            clusters_capacity = []
            print("info.incumbent_interventions:", info.incumbent_interventions)
            for x in info.incumbent_interventions._container[0][1]._container:
                aux = x[1]._container
                if len(aux) > 0:
                    clusters_positions.append(aux[0][1].element_position)
                    clusters_capacity.append(aux[0][1].capacity_to_be_added_MW)

            x1 = len(test_case)-1
            while test_case[x1] != '.':
                x1 -= 1
            x2 = x1-1
            while test_case[x2] != '/' and test_case[x2] != '\\':
                x2 -= 1
            case_name = test_case[x2+1:x1]
            save_in_jsonW(solution, output_dir, NoLines, clusters_positions,
                        clusters_capacity, case_name, yrs, NoScens)

            end = time.time()
            print('\nTime required by the tool:', end - start)


def attest_invest_path(kwargs):
    '''Call ATTEST's distribution network planning tool with paths '''
    input_dir = kwargs.pop('input_dir')
    output_dir = kwargs.pop('output_dir')
    numlines = kwargs.pop('numlines')

    print('\nOptimising investment strategies\n')
    start = time.time()
    solution, info = main_access_function(file_path=input_dir)

    # Get clusters
    clusters_positions = []
    clusters_capacity = []
    for x in info.incumbent_interventions._container[0][1]._container:
        aux = x[1]._container
        if len(aux) > 0:
            clusters_positions.append(aux[0][1].element_position)
            clusters_capacity.append(aux[0][1].capacity_to_be_added_MW)

    x1 = len(input_dir)-1
    while input_dir[x1] != '.':
        x1 -= 1
    x2 = x1-1
    while input_dir[x2] != '/' and input_dir[x2] != '\\':
        x2 -= 1
    case_name = input_dir[x2+1:x1]
    save_in_jsonW(solution, output_dir, numlines, clusters_positions,
                  clusters_capacity, case_name)

    end = time.time()
    print('\nTime required by the tool:', end - start)
