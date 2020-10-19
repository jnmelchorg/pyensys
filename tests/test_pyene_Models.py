""" Test the pyeneE engine. """
from pyene.fixtures import testConfig, json_directory
from pyene.engines.pyene import pyeneClass as pe
import os
import numpy as np

# Network simulation with GLPK
def test_OPF_GLPK():
    """ Verification of results of OPF solution in pyene_Models """
    print('test GLPK with case 4 modified - case4m1.json')
    conf = testConfig()
    conf.NM.settings['File'] = \
        os.path.join(json_directory(), 'case4m1.json')
    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True

    # Hydropower generators
    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [700, 400]  # Generation capacity
    conf.NM.hydropower['Cost'] = [1, 1]  # Costs
    conf.HM.settings['Flag'] = False

    from pyene.engines.pyeneN import ENetworkClass as dn  # Network cengine
    from pyene.engines.pyeneR import RESprofiles as rn  # RES engine

    NM = dn(conf.NM)
    # Initialise
    NM.initialise(rn(conf.RM))

    from pyene.engines.pyene_Models import Networkmodel as NMod # Energy model in glpk
    Model = NMod(NM)
    Model.optimisationNM(solver_name="GLPK")
    
    assert abs(Model.GetObjectiveFunctionNM()-53600) <= 1e-4

    ThermalGeneration = Model.GetThermalGeneration()
    HydroGeneration = Model.GetHydroGeneration()
    VoltageAngle = Model.GetVoltageAngle()
    ActivePowerFlow = Model.GetActivePowerFlow()
 
    thermal_gen_sol = [0.0000, 10.0000, 0.0000, 0.0000, 150.0000, 240.0000]
    hydro_gen_sol = [700, 400]

    power_flow_sol =    [-40.0000, 150.0000, 150.0000, 150.0000, 150.0000, 0.0000, 
                        -150.0000, 0.0000, 0.0000]
    
    vol_angles = [0.0000, 0.0202, -0.0558, 0.0760]

    for xh in Model.LongTemporalConnections:
        for xn in range(Model.NumberConvGen):
            for xt in range(Model.ShortTemporalConnections):
                assert abs(ThermalGeneration[xh, xt, xn] - thermal_gen_sol[xn]) <= 1e-4
        for xn in range(Model.NumberHydroGen):
            for xt in range(Model.ShortTemporalConnections):
                assert abs(HydroGeneration[xh, xt, xn] - hydro_gen_sol[xn]) <= 1e-4
        
        for xb in range(Model.NumberLinesPS):
            for xco in range(Model.NumberContingencies + 1):
                for xt in range(Model.ShortTemporalConnections):
                    assert abs(ActivePowerFlow[xh, xt, xco, xb] - power_flow_sol[xb]) <= 1e-4

        for xn in range(Model.NumberNodesPS):
            for xco in range(Model.NumberContingencies + 1):
                for xt in range(Model.ShortTemporalConnections):
                    assert abs(VoltageAngle[xh, xt, xco, xn] - vol_angles[xn]) <= 1e-4

def test_OPF_GLPK_demC():
    """ Verification of results of OPF solution in pyene_Models for demand curtailment """
    print('test GLPK with case 4 modified - case4m1.json')
    conf = testConfig()
    conf.NM.settings['File'] = \
        os.path.join(json_directory(), 'case4m1.json')
    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True

    # Hydropower generators
    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [200, 100]  # Generation capacity
    conf.NM.hydropower['Cost'] = [1, 1]  # Costs
    conf.HM.settings['Flag'] = False

    from pyene.engines.pyeneN import ENetworkClass as dn  # Network cengine
    from pyene.engines.pyeneR import RESprofiles as rn  # RES engine

    NM = dn(conf.NM)
    # Initialise
    NM.initialise(rn(conf.RM))

    from pyene.engines.pyene_Models import Networkmodel as NMod # Energy model in glpk
    Model = NMod(NM)
    Model.optimisationNM(solver_name="GLPK")
    
    ThermalGeneration = Model.GetThermalGeneration()
    RESGeneration = Model.GetRESGeneration()
    HydroGeneration = Model.GetHydroGeneration()
    if not Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentSystemED()
    elif Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentNodes()
    ActivePowerFlow = Model.GetActivePowerFlow()
    
    thermal_gen_sol = [0.0000, 300.0000, 0.0000, 0.0000, 150.0000, 240.0000]
    hydro_gen_sol = [200, 100]
    total_demand_curtailment = 510

    total_gen_sol = 0
    total_dem_cur_sol = 0

    for xh in Model.LongTemporalConnections:
        for xn in range(Model.NumberConvGen):
            for xt in range(Model.ShortTemporalConnections):
                total_gen_sol += ThermalGeneration[xh, xt, xn]
                assert abs(ThermalGeneration[xh, xt, xn] - thermal_gen_sol[xn]) <= 1e-4
        for xn in range(Model.NumberHydroGen):
            for xt in range(Model.ShortTemporalConnections):
                total_gen_sol += HydroGeneration[xh, xt, xn]
                assert abs(HydroGeneration[xh, xt, xn] - hydro_gen_sol[xn]) <= 1e-4

        for xn in range(Model.NumberNodesPS):
            for xco in range(Model.NumberContingencies + 1):
                for xt in range(Model.ShortTemporalConnections):
                    total_dem_cur_sol += LoadCurtailment[xh, xt, xco, xn]
    
    assert abs(total_dem_cur_sol - 510) <= 1e-4
    assert abs(total_gen_sol - 990) <= 1e-4

def test_OPF_GLPK_genC():
    """ Verification of results of OPF solution in pyene_Models for demand curtailment """
    print('test GLPK with case 4 modified - case4m2.json')
    conf = testConfig()
    conf.NM.settings['File'] = \
        os.path.join(json_directory(), 'case4m2.json')
    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True

    # Hydropower generators
    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [700, 700]  # Generation capacity
    conf.NM.hydropower['Min'] = [0, 600]  # Generation capacity
    conf.NM.hydropower['Cost'] = [1, 1]  # Costs
    conf.HM.settings['Flag'] = False

    from pyene.engines.pyeneN import ENetworkClass as dn  # Network cengine
    from pyene.engines.pyeneR import RESprofiles as rn  # RES engine

    NM = dn(conf.NM)
    # Initialise
    NM.initialise(rn(conf.RM))

    from pyene.engines.pyene_Models import Networkmodel as NMod # Energy model in glpk
    Model = NMod(NM)
    Model.optimisationNM(solver_name="GLPK")
    
    ThermalGeneration = Model.GetThermalGeneration()
    HydroGeneration = Model.GetHydroGeneration()
    GenerationCurtailment = Model.GetGenerationCurtailmentNodes()
    
    thermal_gen_sol = [0.0000, 0.0000, 0.0000, 0.0000, 140.0000, 290.0000]
    hydro_gen_sol = [600, 600]
    gen_cur_sol = [0, 90, 0, 40]

    total_gen_sol = 0
    total_gen_cur_sol = 0

    for xh in Model.LongTemporalConnections:
        for xn in range(Model.NumberConvGen):
            for xt in range(Model.ShortTemporalConnections):
                total_gen_sol += ThermalGeneration[xh, xt, xn]
                assert abs(ThermalGeneration[xh, xt, xn] - thermal_gen_sol[xn]) <= 1e-4
        for xn in range(Model.NumberHydroGen):
            for xt in range(Model.ShortTemporalConnections):
                total_gen_sol += HydroGeneration[xh, xt, xn]
                assert abs(HydroGeneration[xh, xt, xn] - hydro_gen_sol[xn]) <= 1e-4

        for xn in range(Model.NumberNodesPS):
            for xco in range(Model.NumberContingencies + 1):
                for xt in range(Model.ShortTemporalConnections):
                    total_gen_cur_sol += GenerationCurtailment[xh, xt, xco, xn]
                    assert abs(GenerationCurtailment[xh, xt, xco, xn] - gen_cur_sol[xn]) <= 1e-4
    
    assert abs(total_gen_sol - 1500 - total_gen_cur_sol) <= 1e-4

def test_ETree_OPF_GLPK():
    """ Verification of results of energy and network model solution in pyene_Models """
    print('test GLPK with case 4 modified - case4m1.json and tree ResolutionTreeWeek')
    conf = testConfig()
    conf.EM.settings['File'] = os.path.join(json_directory(),
        'ResolutionTreeWeek.json')
    conf.NM.settings['File'] = \
        os.path.join(json_directory(), 'case4m1.json')

    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True
    conf.NM.settings['NoTime'] = 1
    conf.NM.scenarios['NoDem'] = 2

    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [700, 400]  # Generation capacity
    conf.NM.hydropower['Cost'] = [1, 1]  # Costs

    conf.HM.settings['Flag'] = False
    # Create object
    EN = pe(conf.EN)
    # Initialise with selected configuration
    EN.initialise(conf)
    
    EN.set_Demand(1, [0.7])
    EN.set_Demand(2, [1])
    
    EN.set_Hydro(1, 1820)
    EN.set_Hydro(2, 10000)

    from pyene.engines.pyene_Models import EnergyandNetwork as ENMod # Energy model in glpk

    Model = ENMod(EN.EM, EN.NM, EN)
    Model.optimisationENM(solver_name="GLPK")
    
    assert abs(Model.GetObjectiveFunctionENM()-678620) <= 1e-3

    ThermalGeneration = Model.GetThermalGeneration()
    HydroGeneration = Model.GetHydroGeneration()
    GenerationCurtailment = Model.GetGenerationCurtailmentNodes()
    VoltageAngle = Model.GetVoltageAngle()

    tot_demand = [1050, 1500]

    tot_thermal_gen_sol = [0, 0]

    tot_hydro_gen_sol = [0, 0]

    for xh in Model.LongTemporalConnections:
        for xn in range(Model.NumberConvGen):
            for xt in range(Model.ShortTemporalConnections):
                tot_thermal_gen_sol[xh] += ThermalGeneration[xh, xt, xn]
        for xn in range(Model.NumberHydroGen):
            for xt in range(Model.ShortTemporalConnections):
                tot_hydro_gen_sol[xh] += HydroGeneration[xh, xt, xn]

    for xh in Model.LongTemporalConnections:
        assert abs(tot_thermal_gen_sol[xh]+tot_hydro_gen_sol[xh]-tot_demand[xh]) <= 1e-4

    TotalStorage = Model.GetTotalStorage()

    TS_sol =  [0, 7200]
    
    for xv in range(Model.NumberTrees):
        assert abs(TotalStorage[xv, 3] - TS_sol[xv]) <= 1e-4

def test_ETree_OPF_GLPK_hours():
    """ Verification of results of energy and network model solution in pyene_Models with GLPK 
        using 2 representative days and 24h for each representative day"""
    print('test GLPK with case 4 modified - case4m1.json and tree ResolutionTreeWeek')
    conf = testConfig()
    conf.EM.settings['File'] = os.path.join(json_directory(),
        'ResolutionTreeWeek.json')
    conf.NM.settings['File'] = \
        os.path.join(json_directory(), 'case4m1.json')

    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True
    conf.NM.settings['NoTime'] = 24  # Number of time steps
    conf.NM.scenarios['Weights'] = [1 for _ in range(24)]  # Add weights 
        # to the time steps
    # Scenarios
    conf.NM.scenarios['NoDem'] = 2  # Number of demand profiles

    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [700, 400]  # Generation capacity
    conf.NM.hydropower['Cost'] = [1, 1]  # Costs
    
    conf.HM.settings['Flag'] = False
    conf.NM.scenarios['NoRES'] = 0  # Number of RES profiles
    # Create object
    EN = pe(conf.EN)
    # Initialise with selected configuration
    EN.initialise(conf)
    # DEMAND PROFILES
    
    Dem1 = [0.4819, 0.5310, 0.5014, 0.4587, 0.4280, 0.4282, 0.5326, 0.7258, 
        0.8000, 0.8058, 0.7997, 0.7930, 0.7980, 0.7516, 0.7371, 0.7653, 
        0.9142, 1.0000, 0.9401, 0.8702, 0.7960, 0.7441, 0.6415, 0.4969]
    Dem2 = [0.5272, 0.5521, 0.5076, 0.4580, 0.4224, 0.4059, 0.4369, 0.4927, 
        0.5804, 0.6698, 0.7147, 0.7312, 0.7464, 0.6920, 0.6365, 0.6475, 
        0.7496, 0.8689, 0.8854, 0.8421, 0.7675, 0.7148, 0.6274, 0.5115]

    EN.set_Demand(1, Dem1)
    EN.set_Demand(2, Dem2)
    
    EN.set_Hydro(1, 40000)
    EN.set_Hydro(2, 100000)

    from pyene.engines.pyene_Models import EnergyandNetwork as ENMod # Energy model in glpk

    Model = ENMod(EN.EM, EN.NM, EN)
    Model.optimisationENM(solver_name="GLPK")
    
    assert abs(Model.GetObjectiveFunctionENM()-10771254.55) <= 1e-3

    ThermalGeneration = Model.GetThermalGeneration()
    HydroGeneration = Model.GetHydroGeneration()
    if not Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentSystemED()
    elif Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentNodes()
    GenerationCurtailment = Model.GetGenerationCurtailmentNodes()
    VoltageAngle = Model.GetVoltageAngle()

    tot_demand = [25111.65, 22782.75]

    tot_thermal_gen_sol = [0, 0]

    tot_hydro_gen_sol = [0, 0]


    tot_demand_hours = [[722.8499999999999, 796.5, 752.0999999999999, 688.05, 642.0, 642.3, 798.9, 1088.7, 1200.0, 1208.6999999999998, 1199.5499999999997, 1189.5, 1197.0, 1127.4, 1105.6499999999999, 1147.95, 1371.3000000000002, 1500.0, 1410.15, 1305.3, 1194.0, 1116.15, 962.25, 745.3499999999999],
                        [790.8, 828.1500000000001, 761.4000000000001, 686.9999999999999, 633.5999999999999, 608.8499999999999, 655.35, 739.05, 870.5999999999999, 1004.6999999999999, 1072.05, 1096.8, 1119.6, 1037.9999999999998, 954.7499999999999, 971.2499999999999, 1124.4, 1303.35, 1328.1, 1263.1499999999999, 1151.25, 1072.1999999999998, 941.0999999999999, 767.2499999999999]]
    tot_thermal_gen_sol_hours = [[0 for _ in range(24)], [0 for _ in range(24)]]

    tot_hydro_gen_sol_hours = [[0 for _ in range(24)], [0 for _ in range(24)]]

    for xh in Model.LongTemporalConnections:
        for xn in range(Model.NumberConvGen):
            for xt in range(Model.ShortTemporalConnections):
                tot_thermal_gen_sol[xh] += ThermalGeneration[xh, xt, xn]
                tot_thermal_gen_sol_hours[xh][xt] += ThermalGeneration[xh, xt, xn]
        for xn in range(Model.NumberHydroGen):
            for xt in range(Model.ShortTemporalConnections):
                tot_hydro_gen_sol[xh] += HydroGeneration[xh, xt, xn]
                tot_hydro_gen_sol_hours[xh][xt] += HydroGeneration[xh, xt, xn]
        
    for xh in Model.LongTemporalConnections:
        assert abs(tot_thermal_gen_sol[xh]+tot_hydro_gen_sol[xh]-tot_demand[xh]) <= 1e-2
    
    for xh in Model.LongTemporalConnections:
        for xt in range(Model.ShortTemporalConnections):
            assert abs(tot_thermal_gen_sol_hours[xh][xt]+\
                tot_hydro_gen_sol_hours[xh][xt]-tot_demand_hours[xh][xt]) <= 1e-2

    TotalStorage = Model.GetTotalStorage()

    TS_sol =  [0, 32800]
    
    for xv in range(Model.NumberTrees):
        assert abs(TotalStorage[xv, 3] - TS_sol[xv]) <= 1e-4

# Network simulation with CLP
def test_OPF_CLP():
    """ Verification of results of OPF solution in pyene_Models """
    print('test CLP with case 4 modified - case4m1.json')
    conf = testConfig()
    conf.NM.settings['File'] = \
        os.path.join(json_directory(), 'case4m1.json')
    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True

    # Hydropower generators
    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [700, 400]  # Generation capacity
    conf.NM.hydropower['Cost'] = [1, 1]  # Costs
    conf.HM.settings['Flag'] = False

    from pyene.engines.pyeneN import ENetworkClass as dn  # Network cengine
    from pyene.engines.pyeneR import RESprofiles as rn  # RES engine

    NM = dn(conf.NM)
    # Initialise
    NM.initialise(rn(conf.RM))

    from pyene.engines.pyene_Models import Networkmodel as NMod # Energy model in glpk
    Model = NMod(NM)
    Model.optimisationNM(solver_name="CLP")
    
    assert abs(Model.GetObjectiveFunctionNM()-53600) <= 1e-4

    ThermalGeneration = Model.GetThermalGeneration()
    RESGeneration = Model.GetRESGeneration()
    HydroGeneration = Model.GetHydroGeneration()
    PumpOperation = Model.GetPumpOperation()
    if not Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentSystemED()
    elif Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentNodes()
    GenerationCurtailment = Model.GetGenerationCurtailmentNodes()
    VoltageAngle = Model.GetVoltageAngle()
    ActivePowerFlow = Model.GetActivePowerFlow()
    if Model.FlagProblem and Model.LossesFlag:
        ActivePowerLosses = Model.GetActivePowerLosses()
    elif not Model.LossesFlag and Model.PercentageLosses is not None and \
        Model.FlagProblem:
        # Interpolation of losses
        ActivePowerLosses = \
            np.empty((len(Model.LongTemporalConnections),\
                Model.ShortTemporalConnections, \
                (Model.NumberContingencies + 1), \
                Model.NumberLinesPS))
        for xh in Model.LongTemporalConnections:
            for xt in range(Model.ShortTemporalConnections):
                FullLoss = 0
                # Add all power generation
                if Model.NumberConvGen > 0:
                    for xn in range(Model.NumberConvGen):
                        FullLoss += ThermalGeneration[xh, xt, xn]
                if Model.NumberRESGen > 0:
                    for xn in range(Model.NumberRESGen):
                        FullLoss += RESGeneration[xh, xt, xn]
                if Model.NumberHydroGen > 0:
                    for xn in range(Model.NumberHydroGen):
                        FullLoss += HydroGeneration[xh, xt, xn]
                                    
                # Substract all power generation curtailment
                if Model.NumberConvGen > 0:
                    for xn in range(Model.NumberConvGen):
                        for xco in range(Model.NumberContingencies + 1):
                            FullLoss -= ThermalGenerationCurtailment\
                                [xh, xt, xco, xn]
                if Model.NumberRESGen > 0:
                    for xn in range(Model.NumberRESGen):
                        for xco in range(Model.NumberContingencies + 1):
                            FullLoss -= RESGenerationCurtailment\
                                [xh, xt, xco, xn]
                if Model.NumberHydroGen > 0:
                    for xn in range(Model.NumberHydroGen):
                        for xco in range(Model.NumberContingencies + 1):
                            FullLoss -= HydroGenerationCurtailment\
                                [xh, xt, xco, xn]
                # Substract demand
                for xn in range(Model.NumberNodesPS):
                    if Model.NumberDemScenarios == 0:
                        FullLoss -= Model.PowerDemandNode[xn] * \
                            Model.MultScenariosDemand[xh, xn] * \
                            Model.BaseUnitPower
                    else:
                        FullLoss -= Model.PowerDemandNode[xn] * \
                            Model.MultScenariosDemand[xh, xt, xn] * \
                            Model.BaseUnitPower
                    # Curtailment
                    if Model.FlagFeasibility:
                        # Add load curtailment
                        for xco in range(Model.NumberContingencies + 1):
                            FullLoss += LoadCurtailment[xh, xt, xco, xn]
                # Substract non-technical losses
                for xb in range(Model.NumberLinesPS):
                    FullLoss -= Model.NontechnicalLosses[xb] * Model.BaseUnitPower
                # Allocate losses per line
                FullFlow = 0
                for xb in range(Model.NumberLinesPS):
                    for xco in range(Model.NumberContingencies + 1):
                        FullFlow += abs(ActivePowerFlow[xh, xt, xco, xb])
                if FullFlow > 0:
                    for xb in range(Model.NumberLinesPS):
                        for xco in range(Model.NumberContingencies + 1):
                            aux = abs(ActivePowerFlow[xh, xt, xco, xb]) / FullFlow
                        ActivePowerLosses[xh, xt, xco, xb] = FullLoss * aux + \
                            Model.NontechnicalLosses[xb] * Model.BaseUnitPower
    else:
        ActivePowerLosses = \
            np.zeros((len(Model.LongTemporalConnections),\
                Model.ShortTemporalConnections, \
                (Model.NumberContingencies + 1), \
                Model.NumberLinesPS))

    thermal_gen_sol = [0.0000, 10.0000, 0.0000, 0.0000, 150.0000, 240.0000]
    hydro_gen_sol = [700, 400]

    power_flow_sol =    [-40.0000, 150.0000, 150.0000, 150.0000, 150.0000, 0.0000, 
                        -150.0000, 0.0000, 0.0000]
    
    vol_angles = [0.0000, 0.0202, -0.0558, 0.0760]

    losses = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    demand_curtailment = [0.0000, 0.0000, 0.0000, 0.0000]

    for xh in Model.LongTemporalConnections:
        for xn in range(Model.NumberConvGen):
            for xt in range(Model.ShortTemporalConnections):
                assert abs(ThermalGeneration[xh, xt, xn] - thermal_gen_sol[xn]) <= 1e-4
        for xn in range(Model.NumberHydroGen):
            for xt in range(Model.ShortTemporalConnections):
                assert abs(HydroGeneration[xh, xt, xn] - hydro_gen_sol[xn]) <= 1e-4
        
        for xb in range(Model.NumberLinesPS):
            for xco in range(Model.NumberContingencies + 1):
                for xt in range(Model.ShortTemporalConnections):
                    assert abs(ActivePowerFlow[xh, xt, xco, xb] - power_flow_sol[xb]) <= 1e-4

        for xn in range(Model.NumberNodesPS):
            for xco in range(Model.NumberContingencies + 1):
                for xt in range(Model.ShortTemporalConnections):
                    assert abs(VoltageAngle[xh, xt, xco, xn] - vol_angles[xn]) <= 1e-4
        
        for xb in range(Model.NumberLinesPS):
            for xco in range(Model.NumberContingencies + 1):
                for xt in range(Model.ShortTemporalConnections):
                    assert abs(ActivePowerLosses[xh, xt, xco, xb] - losses[xb]) <= 1e-4


        for xn in range(Model.NumberNodesPS):
            for xco in range(Model.NumberContingencies + 1):
                for xt in range(Model.ShortTemporalConnections):
                    assert abs(LoadCurtailment[xh, xt, xco, xn] - demand_curtailment[xn]) <= 1e-4

            # for xn in range(Model.NumberConvGen):
            #     for xco in range(Model.NumberContingencies + 1):
            #         for xt in range(Model.ShortTemporalConnections):
            #             if not Model.FlagFeasibility and \
            #                 Model.NumberConvGen > 0:
            #                 aux = 0
            #             else:
            #                 aux = ThermalGenerationCurtailment\
            #                     [xh, xt, xco, xn]

def test_ETree_OPF_CLP():
    """ Verification of results of energy and network model solution in pyene_Models """
    print('test CLP with case 4 modified - case4m1.json and tree ResolutionTreeWeek')
    conf = testConfig()
    conf.EM.settings['File'] = os.path.join(json_directory(),
        'ResolutionTreeWeek.json')
    conf.NM.settings['File'] = \
        os.path.join(json_directory(), 'case4m1.json')

    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True
    
    conf.NM.settings['NoTime'] = 1
    conf.NM.scenarios['NoDem'] = 2

    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [700, 400]  # Generation capacity
    conf.NM.hydropower['Cost'] = [1, 1]  # Costs

    conf.HM.settings['Flag'] = False
    conf.NM.scenarios['NoRES'] = 0  # Number of RES profiles
    # Create object
    EN = pe(conf.EN)
    # Initialise with selected configuration
    EN.initialise(conf)
    
    EN.set_Demand(1, [0.7])
    EN.set_Demand(2, [1])
    
    EN.set_Hydro(1, 1820)
    EN.set_Hydro(2, 10000)

    from pyene.engines.pyene_Models import EnergyandNetwork as ENMod # Energy model in glpk

    Model = ENMod(EN.EM, EN.NM, EN)
    Model.optimisationENM(solver_name="CLP")

    from pyene.engines.pyeneO import PrintinScreen

    PiS = PrintinScreen(EN)
    PiS.PrintallResults(Model)
    
    assert abs(Model.GetObjectiveFunctionENM()-678620) <= 1e-3

    ThermalGeneration = Model.GetThermalGeneration()
    HydroGeneration = Model.GetHydroGeneration()
    if not Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentSystemED()
    elif Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentNodes()
    GenerationCurtailment = Model.GetGenerationCurtailmentNodes()
    VoltageAngle = Model.GetVoltageAngle()

    tot_demand = [1050, 1500]

    tot_thermal_gen_sol = [0, 0]

    tot_hydro_gen_sol = [0, 0]

    tot_demand_curtailment_sol =  [0, 0]
    for xh in Model.LongTemporalConnections:
        for xn in range(Model.NumberConvGen):
            for xt in range(Model.ShortTemporalConnections):
                tot_thermal_gen_sol[xh] += ThermalGeneration[xh, xt, xn]
        for xn in range(Model.NumberHydroGen):
            for xt in range(Model.ShortTemporalConnections):
                tot_hydro_gen_sol[xh] += HydroGeneration[xh, xt, xn]
        for xn in range(Model.NumberNodesPS):
            for xco in range(Model.NumberContingencies + 1):
                for xt in range(Model.ShortTemporalConnections):
                    tot_demand_curtailment_sol[xh] += LoadCurtailment[xh, xt, xco, xn]
        
    for xh in Model.LongTemporalConnections:
        assert abs(tot_thermal_gen_sol[xh]+tot_hydro_gen_sol[xh]-tot_demand[xh]+\
            tot_demand_curtailment_sol[xh]) <= 1e-4

    TotalStorage = Model.GetTotalStorage()

    TS_sol =  [0, 7200]
    
    for xv in range(Model.NumberTrees):
        assert abs(TotalStorage[xv, 3] - TS_sol[xv]) <= 1e-4

def test_ETree_OPF_CLP_hours():
    """ Verification of results of energy and network model solution in pyene_Models with GLPK 
        using 2 representative days and 24h for each representative day"""
    print('test GLPK with case 4 modified - case4m1.json and tree ResolutionTreeWeek')
    conf = testConfig()
    conf.EM.settings['File'] = os.path.join(json_directory(),
        'ResolutionTreeWeek.json')
    conf.NM.settings['File'] = \
        os.path.join(json_directory(), 'case4m1.json')

    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True
    conf.NM.settings['NoTime'] = 24  # Number of time steps
    conf.NM.scenarios['Weights'] = [1 for _ in range(24)]  # Add weights 
        # to the time steps
    # Scenarios
    conf.NM.scenarios['NoDem'] = 2  # Number of demand profiles

    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [700, 400]  # Generation capacity
    conf.NM.hydropower['Cost'] = [1, 1]  # Costs
    
    conf.HM.settings['Flag'] = False
    conf.NM.scenarios['NoRES'] = 0  # Number of RES profiles
    # Create object
    EN = pe(conf.EN)
    # Initialise with selected configuration
    EN.initialise(conf)
    # DEMAND PROFILES
    
    Dem1 = [0.4819, 0.5310, 0.5014, 0.4587, 0.4280, 0.4282, 0.5326, 0.7258, 
        0.8000, 0.8058, 0.7997, 0.7930, 0.7980, 0.7516, 0.7371, 0.7653, 
        0.9142, 1.0000, 0.9401, 0.8702, 0.7960, 0.7441, 0.6415, 0.4969]
    Dem2 = [0.5272, 0.5521, 0.5076, 0.4580, 0.4224, 0.4059, 0.4369, 0.4927, 
        0.5804, 0.6698, 0.7147, 0.7312, 0.7464, 0.6920, 0.6365, 0.6475, 
        0.7496, 0.8689, 0.8854, 0.8421, 0.7675, 0.7148, 0.6274, 0.5115]

    EN.set_Demand(1, Dem1)
    EN.set_Demand(2, Dem2)
    
    EN.set_Hydro(1, 40000)
    EN.set_Hydro(2, 100000)

    from pyene.engines.pyene_Models import EnergyandNetwork as ENMod # Energy model in glpk

    Model = ENMod(EN.EM, EN.NM, EN)
    Model.optimisationENM(solver_name="CLP")
    
    assert abs(Model.GetObjectiveFunctionENM()-10771254.55) <= 1e-3

    ThermalGeneration = Model.GetThermalGeneration()
    HydroGeneration = Model.GetHydroGeneration()
    if not Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentSystemED()
    elif Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentNodes()
    GenerationCurtailment = Model.GetGenerationCurtailmentNodes()
    VoltageAngle = Model.GetVoltageAngle()

    tot_demand = [25111.65, 22782.75]

    tot_thermal_gen_sol = [0, 0]

    tot_hydro_gen_sol = [0, 0]


    tot_demand_hours = [[722.8499999999999, 796.5, 752.0999999999999, 688.05, 642.0, 642.3, 798.9, 1088.7, 1200.0, 1208.6999999999998, 1199.5499999999997, 1189.5, 1197.0, 1127.4, 1105.6499999999999, 1147.95, 1371.3000000000002, 1500.0, 1410.15, 1305.3, 1194.0, 1116.15, 962.25, 745.3499999999999],
                        [790.8, 828.1500000000001, 761.4000000000001, 686.9999999999999, 633.5999999999999, 608.8499999999999, 655.35, 739.05, 870.5999999999999, 1004.6999999999999, 1072.05, 1096.8, 1119.6, 1037.9999999999998, 954.7499999999999, 971.2499999999999, 1124.4, 1303.35, 1328.1, 1263.1499999999999, 1151.25, 1072.1999999999998, 941.0999999999999, 767.2499999999999]]
    tot_thermal_gen_sol_hours = [[0 for _ in range(24)], [0 for _ in range(24)]]

    tot_hydro_gen_sol_hours = [[0 for _ in range(24)], [0 for _ in range(24)]]

    for xh in Model.LongTemporalConnections:
        for xn in range(Model.NumberConvGen):
            for xt in range(Model.ShortTemporalConnections):
                tot_thermal_gen_sol[xh] += ThermalGeneration[xh, xt, xn]
                tot_thermal_gen_sol_hours[xh][xt] += ThermalGeneration[xh, xt, xn]
        for xn in range(Model.NumberHydroGen):
            for xt in range(Model.ShortTemporalConnections):
                tot_hydro_gen_sol[xh] += HydroGeneration[xh, xt, xn]
                tot_hydro_gen_sol_hours[xh][xt] += HydroGeneration[xh, xt, xn]
        
    for xh in Model.LongTemporalConnections:
        assert abs(tot_thermal_gen_sol[xh]+tot_hydro_gen_sol[xh]-tot_demand[xh]) <= 1e-2
    
    for xh in Model.LongTemporalConnections:
        for xt in range(Model.ShortTemporalConnections):
            assert abs(tot_thermal_gen_sol_hours[xh][xt]+\
                tot_hydro_gen_sol_hours[xh][xt]-tot_demand_hours[xh][xt]) <= 1e-2

    TotalStorage = Model.GetTotalStorage()

    TS_sol =  [0, 32800]
    
    for xv in range(Model.NumberTrees):
        assert abs(TotalStorage[xv, 3] - TS_sol[xv]) <= 1e-4

def test_OPF_CLP_demC():
    """ Verification of results of OPF solution in pyene_Models for demand curtailment """
    print('test GLPK with case 4 modified - case4m1.json')
    conf = testConfig()
    conf.NM.settings['File'] = \
        os.path.join(json_directory(), 'case4m1.json')
    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True

    # Hydropower generators
    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [200, 100]  # Generation capacity
    conf.NM.hydropower['Cost'] = [1, 1]  # Costs
    conf.HM.settings['Flag'] = False

    from pyene.engines.pyeneN import ENetworkClass as dn  # Network cengine
    from pyene.engines.pyeneR import RESprofiles as rn  # RES engine

    NM = dn(conf.NM)
    # Initialise
    NM.initialise(rn(conf.RM))

    from pyene.engines.pyene_Models import Networkmodel as NMod # Energy model in glpk
    Model = NMod(NM)
    Model.optimisationNM(solver_name="CLP")
    
    ThermalGeneration = Model.GetThermalGeneration()
    RESGeneration = Model.GetRESGeneration()
    HydroGeneration = Model.GetHydroGeneration()
    if not Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentSystemED()
    elif Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentNodes()
    ActivePowerFlow = Model.GetActivePowerFlow()
    
    thermal_gen_sol = [0.0000, 300.0000, 0.0000, 0.0000, 150.0000, 240.0000]
    hydro_gen_sol = [200, 100]
    total_demand_curtailment = 510

    total_gen_sol = 0
    total_dem_cur_sol = 0

    for xh in Model.LongTemporalConnections:
        for xn in range(Model.NumberConvGen):
            for xt in range(Model.ShortTemporalConnections):
                total_gen_sol += ThermalGeneration[xh, xt, xn]
                assert abs(ThermalGeneration[xh, xt, xn] - thermal_gen_sol[xn]) <= 1e-4
        for xn in range(Model.NumberHydroGen):
            for xt in range(Model.ShortTemporalConnections):
                total_gen_sol += HydroGeneration[xh, xt, xn]
                assert abs(HydroGeneration[xh, xt, xn] - hydro_gen_sol[xn]) <= 1e-4

        for xn in range(Model.NumberNodesPS):
            for xco in range(Model.NumberContingencies + 1):
                for xt in range(Model.ShortTemporalConnections):
                    total_dem_cur_sol += LoadCurtailment[xh, xt, xco, xn]
    
    assert abs(total_dem_cur_sol - 510) <= 1e-4
    assert abs(total_gen_sol - 990) <= 1e-4

def test_OPF_GLPK_genC():
    """ Verification of results of OPF solution in pyene_Models for demand curtailment """
    print('test GLPK with case 4 modified - case4m2.json')
    conf = testConfig()
    conf.NM.settings['File'] = \
        os.path.join(json_directory(), 'case4m2.json')
    conf.NM.settings['Flag'] = True
    conf.NM.settings['Losses'] = False
    conf.NM.settings['Feasibility'] = True

    # Hydropower generators
    conf.NM.hydropower['Number'] = 2  # Number of hydropower plants
    conf.NM.hydropower['Bus'] = [1, 2]  # Location (bus) of hydro
    conf.NM.hydropower['Max'] = [700, 700]  # Generation capacity
    conf.NM.hydropower['Min'] = [0, 600]  # Generation capacity
    conf.NM.hydropower['Cost'] = [1, 1]  # Costs
    conf.HM.settings['Flag'] = False

    from pyene.engines.pyeneN import ENetworkClass as dn  # Network cengine
    from pyene.engines.pyeneR import RESprofiles as rn  # RES engine

    NM = dn(conf.NM)
    # Initialise
    NM.initialise(rn(conf.RM))

    from pyene.engines.pyene_Models import Networkmodel as NMod # Energy model in glpk
    Model = NMod(NM)
    Model.optimisationNM(solver_name="CLP")
    
    ThermalGeneration = Model.GetThermalGeneration()
    HydroGeneration = Model.GetHydroGeneration()
    GenerationCurtailment = Model.GetGenerationCurtailmentNodes()
    
    thermal_gen_sol = [0.0000, 0.0000, 0.0000, 0.0000, 140.0000, 290.0000]
    hydro_gen_sol = [600, 600]
    gen_cur_sol = [0, 90, 0, 40]

    total_gen_sol = 0
    total_gen_cur_sol = 0

    for xh in Model.LongTemporalConnections:
        for xn in range(Model.NumberConvGen):
            for xt in range(Model.ShortTemporalConnections):
                total_gen_sol += ThermalGeneration[xh, xt, xn]
                assert abs(ThermalGeneration[xh, xt, xn] - thermal_gen_sol[xn]) <= 1e-4
        for xn in range(Model.NumberHydroGen):
            for xt in range(Model.ShortTemporalConnections):
                total_gen_sol += HydroGeneration[xh, xt, xn]
                assert abs(HydroGeneration[xh, xt, xn] - hydro_gen_sol[xn]) <= 1e-4

        for xn in range(Model.NumberNodesPS):
            for xco in range(Model.NumberContingencies + 1):
                for xt in range(Model.ShortTemporalConnections):
                    total_gen_cur_sol += GenerationCurtailment[xh, xt, xco, xn]
                    assert abs(GenerationCurtailment[xh, xt, xco, xn] - gen_cur_sol[xn]) <= 1e-4
    
    assert abs(total_gen_sol - 1500 - total_gen_cur_sol) <= 1e-4


