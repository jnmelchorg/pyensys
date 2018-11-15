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
        self.settings['Case'] = 2  # 0: None, 1: Single, 2: Sensitivity study

        self.data = {}
        self.data['name'] = None
        self.data['cost'] = [0, 0, 0, 0, 0]
        self.data['curtailment'] = 0
        self.data['spill'] = 0
        self.data['OF'] = 0
        self.data['pyomodel'] = None

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

        if self.settings['Directory2'] is not None:
            self.settings['Name2'] = os.path.join(self.settings['Directory2'],
                                                  self.settings['Name2'])
            self.file2 = open_file(self.settings['Name2'], mode='a')

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
        curtailment = Float32Col(dflt=1, pos=7)  # short integer
        spill = Float32Col(dflt=1, pos=8)  # short integer

    def Accumulate(self, EN, m):
        ''' Accumulate results '''
        if self.settings['Directory2'] is None:
            return

        for x in range(5):
            aux = [False for x in range(5)]
            aux[x] = True
            self.data['cost'][x] += EN.get_OFparts(m, aux)

        self.data['curtailment'] += EN.get_AllDemandCurtailment(m)
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
        for xh in range(EN.NM.hydropower['Number']):
            HDF5row['location'] = EN.NM.hydropower['Bus'][xh]
            HDF5row['max'] = EN.NM.hydropower['Max'][xh]
            HDF5row['cost'] = EN.NM.hydropower['Cost'][xh]
            HDF5row['link'] = EN.NM.hydropower['Link'][xh]
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
        for xh in range(EN.NM.RES['Number']):
            HDF5row['location'] = EN.NM.RES['Bus'][xh]
            HDF5row['max'] = EN.NM.RES['Max'][xh]
            HDF5row['cost'] = EN.NM.RES['Cost'][xh]
            HDF5row['link'] = EN.NM.RES['Link'][xh]
            HDF5row.append()
        HDF5table.flush()

    def saveSummary(self):
        if self.settings['Directory2'] is None:
            return
        HDF5group = \
            self.file2.create_group(self.file2.root, self.data['name'])
        self.file2.create_array(HDF5group, "OF", self.data['OF'])
        self.file2.create_array(HDF5group, "curtailment",
                                self.data['curtailment'])
        self.file2.create_array(HDF5group, "spill", self.data['spill'])
        self.file2.create_array(HDF5group, "Cost_Component", self.data['cost'])
        self.file2.create_array(HDF5group, "time", [self.time['All'],
                                                    self.time['glpk']])

        self.file2.close()

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
                    EN.NM.networkE.graph['baseMVA']
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
                                                     times=[xt], scens=[xs])
                HDF5row['pump'] = EN.get_AllPumps(m, 'snapshot', times=[xt],
                                                  scens=[xs])
                HDF5row['loss'] = EN.get_AllLoss(m, 'snapshot', times=[xt],
                                                 scens=[xs])
                HDF5row['curtailment'] = \
                    EN.get_AllDemandCurtailment(m, 'snapshot', times=[xt],
                                                scens=[xs])
                HDF5row.append()
            HDF5table.flush()

    def terminate(self):

        if self.settings['Directory1'] is None:
            return
        self.fileh.close()
