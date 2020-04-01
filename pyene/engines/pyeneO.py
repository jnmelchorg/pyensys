# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:32:08 2018

Pyene Outputs provides methods for saving simualtion outputs in HDF5 files

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import numpy as np
from tables import Int16Col, Float32Col, StringCol, IsDescription, open_file
import os


class pyeneOConfig:
    ''' Default settings used for this class '''
    def __init__(self):
        # Default time-step and map
        self.settings = {}
        self.settings['Directory1'] = None  # Location of main folder
        self.settings['Directory2'] = None  # Location of main folder
        self.settings['Name1'] = 'pyeneOutputs.h5'  # Name
        self.settings['Name2'] = 'pyeneSensitivity.h5'  # Name
        self.settings['Name3'] = 'pyeneDetailedOutputs.h5'  # Name
        self.settings['Case'] = 2  # 0: None, 1: Single, 2: Sensitivity study

        self.data = {}
        self.data['name'] = None
        self.data['cost'] = [0, 0, 0, 0, 0]
        self.data['curtailment'] = 0
        self.data['spill'] = 0
        self.data['OF'] = 0
        self.data['pyomodel'] = None
        self.data['GLPKmodel'] = None

        self.time = {}
        self.time['All'] = 0
        self.time['glpk'] = 0
        self.time['step'] = None


class pyeneHDF5Settings():
    def __init__(self, obj=None):
        ''' Initialise network class '''
        # Get default values
        if obj is None:
            obj = pyeneOConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

        # No outputs if there is no output directory
        if self.settings['Directory1'] is not None:
            self.settings['Name1'] = os.path.join(self.settings['Directory1'],
                                                  self.settings['Name1'])
            self.fileh = open_file(self.settings['Name1'], mode='w')
            self.settings['Name3'] = os.path.join(self.settings['Directory1'],
                                                  self.settings['Name3'])
            self.filedetailedinfo = open_file(self.settings['Name3'], mode='w')

        # if self.settings['Directory2'] is not None:
        #     self.settings['Name2'] = os.path.join(self.settings['Directory2'],
        #                                           self.settings['Name2'])
        #     self.file2 = open_file(self.settings['Name2'], mode='a')

    '''pytables auxiliary'''
    class PyeneHDF5Flags:
        def __init__(self):
            self.settings = {}
            self.settings['treefile'] = True
            self.settings['networkfile'] = True
            self.settings['time'] = True
            self.settings['scenarios'] = True
            self.settings['NoHydro'] = True
            self.settings['NoPump'] = True
            self.settings['NoRES'] = True

            self.devices = {}
            self.results = {}

    class PyeneHDF5Settings(IsDescription):
        '''Simulation settings'''
        treefile = StringCol(itemsize=30, dflt=" ", pos=0)  # File name (tree)
        networkfile = StringCol(itemsize=30, dflt=" ", pos=1)  # FIle (network)
        time = Int16Col(dflt=1, pos=2)  # generatop
        scenarios = Int16Col(dflt=1, pos=3)  # scenarios
        NoHydro = Int16Col(dflt=1, pos=4)  # hydropower plants
        NoPump = Int16Col(dflt=1, pos=5)  # pumps
        NoRES = Int16Col(dflt=1, pos=6)  # RES generators

    class PyeneHDF5Devices(IsDescription):
        '''Characteristics of devices '''
        location = Int16Col(dflt=1, pos=0)  # Location
        max = Float32Col(dflt=1, pos=1)  # Capacity
        cost = Float32Col(dflt=4, pos=2)  # Cost/value
        link = Int16Col(dflt=1, pos=3)  # Location

    class PyeneHDF5Results(IsDescription):
        time = Int16Col(dflt=1, pos=0)  # time period
        generation = Float32Col(dflt=1, pos=1)  # conventional generator
        hydropower = Float32Col(dflt=1, pos=2)  # hydropower
        RES = Float32Col(dflt=1, pos=3)  # RES generator
        demand = Float32Col(dflt=1, pos=4)  # total demand
        pump = Float32Col(dflt=1, pos=5)  # use of pumps
        loss = Float32Col(dflt=1, pos=6)  # losses
        curtailment = Float32Col(dflt=1, pos=7)  # sCurtailment
        spill = Float32Col(dflt=1, pos=8)  # spilling

    def Accumulate(self, EN, m):
        ''' Accumulate results '''
        if self.settings['Directory2'] is None:
            return

        for x in range(5):
            aux = [False for x in range(5)]
            aux[x] = True
            self.data['cost'][x] += EN.get_OFparts(m, aux)

        self.data['curtailment'] += EN.get_AllDemandCurtailment(m)[0]
        self.data['spill'] += EN.get_AllRES(m)
        self.data['OF'] += EN.get_OFparts(m, [True for x in range(5)])

    def SaveSettings(self, EN):
        ''' Save simulation settings '''
        if self.settings['Directory1'] is None:
            return

        HDF5group = self.fileh.create_group(self.fileh.root, 'Core')
        HDF5table = self.fileh.create_table(HDF5group, "table",
                                            self.PyeneHDF5Settings)
        HDF5row = HDF5table.row
        HDF5row['treefile'] = EN.EM.settings['File']
        HDF5row['networkfile'] = EN.NM.settings['File']
        HDF5row['time'] = EN.NM.settings['NoTime']
        HDF5row['scenarios'] = EN.NM.scenarios['Number']
        HDF5row['NoHydro'] = EN.NM.hydropower['Number']
        HDF5row['NoPump'] = EN.NM.pumps['Number']
        HDF5row['NoRES'] = EN.NM.RES['Number']
        HDF5row.append()
        HDF5table.flush()

        HDF5table = self.fileh.create_table(HDF5group, "Hydpopower_plants",
                                            self.PyeneHDF5Devices)
        HDF5row = HDF5table.row
        for x in EN.NM.Gen.Hydro:
            HDF5row['location'] = x.get_BusPos()
            HDF5row['max'] = x.get_Max()
            HDF5row['cost'] = x.get_Cost()
            HDF5row.append()
        HDF5table.flush()

        HDF5table = self.fileh.create_table(HDF5group, "Pumps",
                                            self.PyeneHDF5Devices)
        HDF5row = HDF5table.row
        for xh in range(EN.NM.pumps['Number']):
            HDF5row['location'] = EN.NM.pumps['Bus'][xh]
            HDF5row['max'] = EN.NM.pumps['Max'][xh]
            HDF5row['cost'] = EN.NM.pumps['Value'][xh]
            HDF5row.append()
        HDF5table.flush()

        HDF5table = self.fileh.create_table(HDF5group, "RES_generators",
                                            self.PyeneHDF5Devices)
        HDF5row = HDF5table.row
        for x in EN.NM.Gen.RES:
            HDF5row['location'] = x.get_BusPos()
            HDF5row['max'] = x.get_Max()
            HDF5row['cost'] = x.get_Cost()
            HDF5row.append()

        HDF5table.flush()

    def saveSummary(self, simulation_name):
        # Independent files
        if self.settings['Directory1'] is not None:
            aux = os.path.join(self.settings['Directory2'], simulation_name)
            fileh = open_file(aux, mode='w')
            fileh.create_array(fileh.root, "OF", self.data['OF'])
            fileh.create_array(fileh.root, "curtailment",
                               self.data['curtailment'])
            fileh.create_array(fileh.root, "spill", self.data['spill'])
            fileh.create_array(fileh.root, "Cost_Component", self.data['cost'])
            fileh.create_array(fileh.root, "time", [self.time['All'],
                                                    self.time['glpk']])
            fileh.close()

    def saveResults(self, EN, m, SimNo):
        ''' Save results of each iteration '''

        # Accumulate data
        self.Accumulate(EN, m)

        if self.settings['Directory1'] is None:
            return

        HDF5group = \
            self.fileh.create_group(self.fileh.root,
                                    'Simulation_{:05d}'.format(SimNo))
        HDF5aux = np.zeros((EN.NM.scenarios['NoDem'],
                            EN.NM.settings['NoTime']), dtype=float)

        xp = 0
        for xs in range(EN.NM.scenarios['NoDem']):
            for xt in range(EN.NM.settings['NoTime']):
                HDF5aux[xs][xt] = EN.NM.scenarios['Demand'][xp]
                xp += 1

        self.fileh.create_array(HDF5group, "Demand_profiles", HDF5aux)

        HDF5aux = np.zeros((EN.NM.scenarios['NoRES'],
                            EN.NM.settings['NoTime']), dtype=float)
        xp = 0
        for xs in range(EN.NM.scenarios['NoRES']):
            for xt in range(EN.NM.settings['NoTime']):
                HDF5aux[xs][xt] = EN.NM.scenarios['RES'][xp] * \
                    EN.NM.ENetwork.get_Base()
                xp += 1
        self.fileh.create_array(HDF5group, "RES_profiles", HDF5aux)

        # Hydropower allowance
        aux = np.zeros(EN.EM.settings['Vectors'], dtype=float)
        if type(m.vEIn) is np.ndarray:
            if EN.EM.settings['Vectors'] == 1:
                aux[0] = EN.EM.Weight['In'][1]
            else:
                for xv in EN.EM.s['Vec']:
                    aux[xv] = m.vEIn[1][xv]
        else:
            for xv in EN.EM.s['Vec']:
                aux[xv] = m.vEIn[1, xv].value

        self.fileh.create_array(HDF5group, "Hydro_Allowance", aux)

        hp_marginal = np.zeros(EN.EM.settings['Vectors'], dtype=float)
        for xi in range(EN.EM.settings['Vectors']):
            hp_marginal[xi] = EN.get_HydroMarginal(m, xi+1)
        self.fileh.create_array(HDF5group, "Hydro_Marginal", hp_marginal)

        for xs in range(EN.NM.scenarios['Number']):
            HDF5table = \
                self.fileh.create_table(HDF5group,
                                        "Scenario_{:02d}".format(xs),
                                        self.PyeneHDF5Results)
            HDF5row = HDF5table.row
            for xt in range(EN.NM.settings['NoTime']):
                HDF5row['time'] = xt
                HDF5row['generation'] = \
                    EN.get_AllGeneration(m, 'Conv', 'snapshot', times=[xt],
                                         scens=[xs])
                HDF5row['hydropower'] = \
                    EN.get_AllGeneration(m, 'Hydro', 'snapshot', times=[xt],
                                         scens=[xs])
                HDF5row['RES'] = EN.get_AllGeneration(m, 'RES', 'snapshot',
                                                      times=[xt], scens=[xs])
                HDF5row['spill'] = EN.get_AllRES(m, 'snapshot', times=[xt],
                                                 scens=[xs])
                HDF5row['demand'] = EN.get_AllDemand(m, 'snapshot',
                                                     times=[xt], scens=[xs])[0]
                HDF5row['pump'] = EN.get_AllPumps(m, 'snapshot', times=[xt],
                                                  scens=[xs])
                HDF5row['loss'] = EN.get_AllLoss(m, 'snapshot', times=[xt],
                                                 scens=[xs])
                HDF5row['curtailment'] = \
                    EN.get_AllDemandCurtailment(m, 'snapshot', times=[xt],
                                                scens=[xs])[0]
                HDF5row.append()
            HDF5table.flush()

    def saveResultsGLPK(self, EN, GLPKobj, SimNo):
        ''' Save results of each iteration '''

        # Accumulate data
        # self.Accumulate(EN, m)

        if self.settings['Directory1'] is None:
            return

        HDF5group = \
            self.fileh.create_group(self.fileh.root,
                                    'Simulation_{:05d}'.format(SimNo))
        HDF5aux = np.zeros((GLPKobj.NumberDemScenarios,
                            GLPKobj.ShortTemporalConnections), dtype=float)

        xp = 0
        for xs in range(GLPKobj.NumberDemScenarios):
            for xt in range(GLPKobj.ShortTemporalConnections):
                HDF5aux[xs][xt] = EN.NM.scenarios['Demand'][xp]
                xp += 1

        self.fileh.create_array(HDF5group, "Demand_profiles", HDF5aux)

        HDF5aux = np.zeros((EN.NM.scenarios['NoRES'],
                            EN.NM.settings['NoTime']), dtype=float)
        xp = 0
        for xs in range(EN.NM.scenarios['NoRES']):
            for xt in range(EN.NM.settings['NoTime']):
                HDF5aux[xs][xt] = EN.NM.scenarios['RES'][xp] * \
                    EN.NM.ENetwork.get_Base()
                xp += 1
        self.fileh.create_array(HDF5group, "RES_profiles", HDF5aux)

        # Hydropower allowance
        aux = np.zeros(EN.EM.settings['Vectors'], dtype=float)
        if EN.EM.settings['Vectors'] == 1:
            aux[0] = EN.EM.Weight['In'][1]
        else:
            for xv in range(EN.EM.settings['Vectors']):
                aux[xv] = GLPKobj.IntakeTree[1, xv]

        self.fileh.create_array(HDF5group, "Hydro_Allowance", aux)

        hp_marginal = np.zeros(EN.EM.settings['Vectors'], dtype=float)
        for xi in range(EN.EM.settings['Vectors']):
            hp_marginal[xi] = 0.0
        self.fileh.create_array(HDF5group, "Hydro_Marginal", hp_marginal)

        # Getting the solution from GLPK

        ThermalGeneration = GLPKobj.GetThermalGeneration()
        RESGeneration = GLPKobj.GetRESGeneration()
        HydroGeneration = GLPKobj.GetHydroGeneration()
        PumpOperation = GLPKobj.GetPumpOperation()
        if GLPKobj.FlagProblem and GLPKobj.FlagFeasibility:
            LoadCurtailment = GLPKobj.GetLoadCurtailmentNodes()
        else:
            LoadCurtailment = GLPKobj.GetLoadCurtailmentSystemED()
        ActivePowerLosses = GLPKobj.GetActivePowerLosses()
        

        for xs in GLPKobj.LongTemporalConnections:
            HDF5table = \
                self.fileh.create_table(HDF5group,
                                        "Scenario_{:02d}".format(xs),
                                        self.PyeneHDF5Results)
            HDF5row = HDF5table.row
            for xt in range(GLPKobj.ShortTemporalConnections):
                HDF5row['time'] = xt
                auxvar = 0
                if ThermalGeneration is not None:
                    for k in range(GLPKobj.NumberConvGen):
                        auxvar += ThermalGeneration[xs, xt, k]
                HDF5row['generation'] = auxvar

                
                auxvar = 0
                if HydroGeneration is not None:
                    for k in range(GLPKobj.NumberHydroGen):
                        auxvar += HydroGeneration[xs, xt, k]
                HDF5row['hydropower'] = auxvar
                
                auxvar = 0
                if RESGeneration is not None:
                    for k in range(GLPKobj.NumberRESGen):
                        auxvar += RESGeneration[xs, xt, k]
                HDF5row['RES'] = auxvar

                HDF5row['spill'] = 0

                auxvar = 0
                for k in range(EN.NM.ENetwork.get_NoBus()):
                    auxvar += EN.NM.busData[k] * \
                        EN.NM.scenarios['Demand']\
                            [EN.NM.busScenario[k][xs]] * \
                                EN.NM.ENetwork.get_Base()
                HDF5row['demand'] = auxvar

                auxvar = 0
                if PumpOperation is not None:
                    for k in range(GLPKobj.NumberPumps):
                        auxvar += PumpOperation[xs, xt, k]
                HDF5row['pump'] = auxvar

                auxvar = 0
                if EN.NM.settings['Losses']:
                    for k in range(GLPKobj.NumberContingencies + 1):
                        for ii in range(GLPKobj.NumberLinesPS):
                            auxvar += ActivePowerLosses[xs, xt, k]
                HDF5row['loss'] = auxvar

                auxvar = 0
                if GLPKobj.FlagProblem and LoadCurtailment is not None:
                    for k in range(GLPKobj.NumberContingencies + 1):
                        for ii in range(GLPKobj.NumberNodesPS):
                            auxvar += LoadCurtailment[xs, xt, k, ii]
                if not GLPKobj.FlagProblem and LoadCurtailment is not None:
                    auxvar += LoadCurtailment[xs, xt]
                HDF5row['curtailment'] = auxvar
                HDF5row.append()
            HDF5table.flush()

    def SaveDetailedResultsGLPK(self, EN, GLPKobj, SimNo):
        ''' Save results of each iteration '''

        # Accumulate data
        # self.Accumulate(EN, m)

        if self.settings['Directory1'] is None:
            return

        HDF5group = \
            self.filedetailedinfo.create_group(self.filedetailedinfo.root,
                                    'Simulation_{:05d}'.format(SimNo))
        HDF5aux = np.zeros((GLPKobj.NumberDemScenarios,
                            GLPKobj.ShortTemporalConnections), dtype=float)

        xp = 0
        for xs in range(GLPKobj.NumberDemScenarios):
            for xt in range(GLPKobj.ShortTemporalConnections):
                HDF5aux[xs][xt] = EN.NM.scenarios['Demand'][xp]
                xp += 1

        self.filedetailedinfo.create_array(HDF5group, "Demand_profiles", HDF5aux)

        HDF5aux = np.zeros((EN.NM.scenarios['NoRES'],
                            EN.NM.settings['NoTime']), dtype=float)
        xp = 0
        for xs in range(EN.NM.scenarios['NoRES']):
            for xt in range(EN.NM.settings['NoTime']):
                HDF5aux[xs][xt] = EN.NM.scenarios['RES'][xp] * \
                    EN.NM.ENetwork.get_Base()
                xp += 1
        self.filedetailedinfo.create_array(HDF5group, "RES_profiles", HDF5aux)

        # Hydropower allowance
        aux = np.zeros(EN.EM.settings['Vectors'], dtype=float)
        if EN.EM.settings['Vectors'] == 1:
            aux[0] = EN.EM.Weight['In'][1]
        else:
            for xv in range(EN.EM.settings['Vectors']):
                aux[xv] = GLPKobj.IntakeTree[1, xv]

        self.filedetailedinfo.create_array(HDF5group, "Hydro_Allowance", aux)

        hp_marginal = np.zeros(EN.EM.settings['Vectors'], dtype=float)
        for xi in range(EN.EM.settings['Vectors']):
            hp_marginal[xi] = 0.0
        self.filedetailedinfo.create_array(HDF5group, "Hydro_Marginal", hp_marginal)

        # Getting the solution from GLPK

        ThermalGeneration = GLPKobj.GetThermalGeneration()
        RESGeneration = GLPKobj.GetRESGeneration()
        HydroGeneration = GLPKobj.GetHydroGeneration()
        PumpOperation = GLPKobj.GetPumpOperation()
        if GLPKobj.FlagProblem and GLPK.FlagFeasibility:
            LoadCurtailment = GLPKobj.GetLoadCurtailmentSystemED()

        else:
            LoadCurtailment = GLPKobj.GLPKobj.GetLoadCurtailmentNodes()
        ActivePowerLosses = GLPKobj.GetActivePowerLosses()
        VoltageAngle = GLPKobj.GetVoltageAngle()
        ActivePowerFlow = GLPKobj.GetActivePowerFlow()

        if ThermalGeneration is not None:
            self.filedetailedinfo.create_array(HDF5group, "Thermal_Generation", \
                ThermalGeneration)
        
        if RESGeneration is not None:
            self.filedetailedinfo.create_array(HDF5group, "RES_Generation", \
                RESGeneration)
        
        if HydroGeneration is not None:
            self.filedetailedinfo.create_array(HDF5group, "Hydro_Generation", \
                HydroGeneration)
        
        if PumpOperation is not None:
            self.filedetailedinfo.create_array(HDF5group, "Pump_Operation", \
                PumpOperation)
        
        if LoadCurtailment is not None:
            self.filedetailedinfo.create_array(HDF5group, "Load_Curtailment", \
                LoadCurtailment)
        
        if ActivePowerLosses is not None:
            self.filedetailedinfo.create_array(HDF5group, "Active_Power_Losses", \
                ActivePowerLosses)
        
        if VoltageAngle is not None:
            self.filedetailedinfo.create_array(HDF5group, "Voltage_Angle", \
                VoltageAngle)
        
        if ActivePowerFlow is not None:
            self.filedetailedinfo.create_array(HDF5group, "Active_Power_Flow", \
                ActivePowerFlow)



    def terminate(self):

        if self.settings['Directory1'] is None:
            return
        self.fileh.close()
        self.filedetailedinfo.close()

class PrintinScreen():
    '''This class contains all definitions that allows printing on the screen any 
    information requiested by the users'''
    def __init__(self, obj=None):
        if obj is None:
            assert('No information has been passed to this class')
        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))
        self.PrintinScreenOptions = {
                'Generation': True,
                'Flows': True,
                'Voltages': True,
                'Losses': True,
                'Curtailment': True,
                'Feasibility': True,
                'Services': True,
                'GenBus': True,
                'UC': True,
                'SoC': True,
                }
        self.lossesresultsOPF = None
    def printlosses(self):
        if self.PrintinScreenOptions['Losses'] and self.settings['Flag']:
            for xh in sh:
                print("\nEPower_Loss=[")
                LossDt = self.printLosses(m, xh)
                for xb in range(self.ENetwork.get_NoBra()):
                    for xt in self.s['Tim']:
                        print("%8.4f " % LossDt[xb][xt], end='')
                    print()
                print("];")

    def printallNetworkResult(self, obj=None):
        ''' This class method prints on the screen all results for the \
            network model'''
        if obj is None:
            print('No object with GLPK results has been passed - Aborting \
                printing of results')
            return

        Generator = self.NM.Gen
        Branches = self.NM.ENetwork.Branch
        ThermalGeneration = obj.GetThermalGeneration()
        RESGeneration = obj.GetRESGeneration()
        HydroGeneration = obj.GetHydroGeneration()
        PumpOperation = obj.GetPumpOperation()
        if not obj.FlagProblem and obj.FlagFeasibility:
            LoadCurtailment = obj.GetLoadCurtailmentSystemED()
        elif obj.FlagProblem and obj.FlagFeasibility:
            LoadCurtailment = obj.GetLoadCurtailmentNodes()
        ThermalGenerationCurtailment = obj.GetThermalGenerationCurtailmentNodes()
        RESGenerationCurtailment = obj.GetRESGenerationCurtailmentNodes()
        HydroGenerationCurtailment = obj.GetHydroGenerationCurtailmentNodes()
        VoltageAngle = obj.GetVoltageAngle()
        ActivePowerFlow = obj.GetActivePowerFlow()

        if obj.FlagProblem and obj.LossesFlag:
            ActivePowerLosses = obj.GetActivePowerLosses()
        elif not obj.LossesFlag and self.PercentageLosses is not None and \
            obj.FlagProblem:
            # Interpolation of losses
            ActivePowerLosses = \
                np.empty((len(obj.LongTemporalConnections),\
                    obj.ShortTemporalConnections, \
                    (obj.NumberContingencies + 1), \
                    obj.NumberLinesPS))
            for xh in obj.LongTemporalConnections:
                for xt in range(obj.ShortTemporalConnections):
                    FullLoss = 0
                    # Add all power generation
                    if obj.NumberConvGen > 0:
                        for xn in range(obj.NumberConvGen):
                            FullLoss += ThermalGeneration[xh, xt, xn]
                    if obj.NumberRESGen > 0:
                        for xn in range(obj.NumberRESGen):
                            FullLoss += RESGeneration[xh, xt, xn]
                    if obj.NumberHydroGen > 0:
                        for xn in range(obj.NumberHydroGen):
                            FullLoss += HydroGeneration[xh, xt, xn]
                    
                    # Substract all power generation curtailment
                    if obj.NumberConvGen > 0:
                        for xn in range(obj.NumberConvGen):
                            for xco in range(obj.NumberContingencies + 1):
                                FullLoss -= ThermalGenerationCurtailment\
                                    [xh, xt, xco, xn]
                    if obj.NumberRESGen > 0:
                        for xn in range(obj.NumberRESGen):
                            for xco in range(obj.NumberContingencies + 1):
                                FullLoss -= RESGenerationCurtailment\
                                    [xh, xt, xco, xn]
                    if obj.NumberHydroGen > 0:
                        for xn in range(obj.NumberHydroGen):
                            for xco in range(obj.NumberContingencies + 1):
                                FullLoss -= HydroGenerationCurtailment\
                                    [xh, xt, xco, xn]

                    # Substract demand
                    for xn in range(obj.NumberNodesPS):
                        if obj.NumberDemScenarios == 0:
                            FullLoss -= demand.PowerDemandNode[xn] * \
                                obj.MultScenariosDemand[xh, xn] * \
                                (1 + obj.PercentageLosses) * \
                                obj.BaseUnitPower
                        else:
                            FullLoss -= obj.PowerDemandNode[xn] * \
                                obj.MultScenariosDemand[xh, xt, xn] * \
                                (1 + obj.PercentageLosses) * \
                                obj.BaseUnitPower

                        # Curtailment
                        if obj.FlagFeasibility:
                            # Add load curtailment
                            for xco in range(self.NumberContingencies + 1):
                                FullLoss += LoadCurtailment[xh, xt, xco, xn]

                    # Substract non-technical losses
                    for xb in range(obj.NumberLinesPS):
                        FullLoss[xt] -= Branches[xb].getLoss()

                    # Allocate losses per line
                    FullFlow = 0
                    for xb in range(obj.NumberLinesPS):
                        for xco in range(self.NumberContingencies + 1):
                            FullFlow += abs(ActivePowerFlow[xh, xt, xco, xb])
                    if FullFlow > 0:
                        for xb in range(obj.NumberLinesPS):
                            for xco in range(self.NumberContingencies + 1):
                                aux = abs(ActivePowerFlow[xh, xt, xco, xb]) / FullFlow
                            ActivePowerLosses[xh, xt, xco, xb] = FullLoss * aux + \
                                Branches[xb].getLoss()

        # Printing results

        for xh in obj.LongTemporalConnections:
            self.printallEnergyResults(obj)
            print("\n% CASE:", xh)

            if self.PrintinScreenOptions['GenBus']:
                print('\nFlow_EGen_Bus=', Generator.get_GenDataAll(), ';')

            if self.PrintinScreenOptions['Generation']:
                print("\nFlow_EGen=[")
                if obj.NumberConvGen > 0:
                    for xn in range(obj.NumberConvGen):
                        for xt in range(obj.ShortTemporalConnections):
                            print("%8.4f " % ThermalGeneration[xh, xt, xn], \
                                end='')
                        print()
                if obj.NumberRESGen > 0:
                    for xn in range(obj.NumberRESGen):
                        for xt in range(obj.ShortTemporalConnections):
                            print("%8.4f " % RESGeneration[xh, xt, xn], \
                                end='')
                        print()
                if obj.NumberHydroGen > 0:
                    for xn in range(obj.NumberHydroGen):
                        for xt in range(obj.ShortTemporalConnections):
                            print("%8.4f " % HydroGeneration[xh, xt, xn], \
                                end='')
                        print()
                print("];")

            if self.PrintinScreenOptions['UC']:
                print("\nBin_EGen=[")
                aux = 1
                if obj.NumberConvGen > 0:
                    for xn in range(obj.NumberConvGen):
                        for xt in range(obj.ShortTemporalConnections):
                            print("%2.0f " % aux, end='')
                        print()
                if obj.NumberRESGen > 0:
                    for xn in range(obj.NumberRESGen):
                        for xt in range(obj.ShortTemporalConnections):
                            print("%2.0f " % aux, end='')
                        print()
                if obj.NumberHydroGen > 0:
                    for xn in range(obj.NumberHydroGen):
                        for xt in range(obj.ShortTemporalConnections):
                            print("%2.0f " % aux, end='')
                        print()
                print("];")

            if self.PrintinScreenOptions['Flows'] and obj.FlagProblem:
                print("\nFlow_EPower=[")
                for xb in range(obj.NumberLinesPS):
                    for xco in range(obj.NumberContingencies + 1):
                        for xt in range(obj.ShortTemporalConnections):
                            print("%8.4f " % ActivePowerFlow[xh, xt, xco, xb], \
                                end='')
                        print()
                print("];")

            if self.PrintinScreenOptions['Voltages'] and obj.FlagProblem:
                print("\nVoltage_Angle=[")
                for xn in range(obj.NumberNodesPS):
                    for xco in range(obj.NumberContingencies + 1):
                        for xt in range(obj.ShortTemporalConnections):
                            print("%8.4f " % VoltageAngle[xh, xt, xco, xn], \
                                end='')
                        print()
                print("];")

            if self.PrintinScreenOptions['Losses'] and obj.FlagProblem:
                print("\nEPower_Loss=[")
                for xb in range(obj.NumberLinesPS):
                    for xco in range(obj.NumberContingencies + 1):
                        for xt in range(obj.ShortTemporalConnections):
                            print("%8.4f " % ActivePowerLosses[xh, xt, xco, xb]\
                                , end='')
                    print()
                print("];")

            if self.PrintinScreenOptions['Curtailment']:
                print("\nPumps=[")
                if obj.NumberPumps > 0:
                    for xpu in range(obj.NumberPumps):
                        for xt in range(obj.ShortTemporalConnections):
                            print("%8.4f " % PumpOperation[xh, xt, xpu], end='')
                        print()
                print("];")

            if self.PrintinScreenOptions['Feasibility']:
                print("\nFeas=[")
                for xn in range(obj.NumberNodesPS):
                    for xco in range(obj.NumberContingencies + 1):
                        for xt in range(obj.ShortTemporalConnections):
                            if not obj.FlagFeasibility:
                                aux = 0
                            else:
                                aux = LoadCurtailment[xh, xt, xco, xn]
                            print("%8.4f " % aux, end='')
                    print()
                print("];")
            print()

    def printallEnergyResults(self, obj=None):
        ''' This class method prints on the screen all results for the \
            energy model'''
        if obj is None:
            print('No object with GLPK results has been passed - Aborting \
                printing of results')
            return
        
        PartialStorage = obj.GetPartialStorage()
        TotalStorage = obj.GetTotalStorage()

        if self.PrintinScreenOptions['SoC']:
            for xv in range(obj.NumberTrees):
                print('Vector No:', xv)
                for xtr in range(obj.TreeNodes):
                    print("SoC[%3.0f" % xtr, "][0:1]=[%10.2f"
                          % PartialStorage[xv, xtr], ", %10.2f"
                          % TotalStorage[xv, xtr], "]")
    
    #TODO: This is not well printed with plenty of repetitions, to be discussed 
    # with Alex and Paul
    def PrintallResults(self, obj=None):
        ''' This class method prints on the screen all results for the \
            energy and network model'''
        if obj is None:
            print('No object with GLPK results has been passed - Aborting \
                printing of results')
            return
        
        self.printallEnergyResults(obj)
        self.printallNetworkResult(obj)

        InputsTree = obj.GetInputsTree()
        OutputsTree = obj.GetOutputsTree()
        print('Water outputs:')
        for xn in range(obj.TreeNodes):
            for xv in range(obj.NumberTrees):
                print("%8.4f " % OutputsTree[xv, xn], end='')
            print('')
        print('Water inputs:')
        for xn in range(obj.TreeNodes):
            for xv in range(obj.NumberTrees):
                print("%8.4f " % InputsTree[xv, xn], end='')
            print('')