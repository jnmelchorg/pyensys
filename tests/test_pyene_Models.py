""" Test the pyeneE engine. """
from pyene.fixtures import testConfig, json_directory
from pyene.engines.pyene import pyeneClass as pe
import os
import numpy as np

# Network simulation with GLPK
def test_OPF_GLPK():
    """ Verification of results in GLPK """
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
    RESGeneration = Model.GetRESGeneration()
    HydroGeneration = Model.GetHydroGeneration()
    PumpOperation = Model.GetPumpOperation()
    if not Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentSystemED()
    elif Model.FlagProblem and Model.FlagFeasibility:
        LoadCurtailment = Model.GetLoadCurtailmentNodes()
    ThermalGenerationCurtailment = Model.GetThermalGenerationCurtailmentNodes()
    RESGenerationCurtailment = Model.GetRESGenerationCurtailmentNodes()
    HydroGenerationCurtailment = Model.GetHydroGenerationCurtailmentNodes()
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
