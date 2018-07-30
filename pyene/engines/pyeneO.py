# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:32:08 2018

Pyene Outputs provides methods for saving simualtion outputs in HDF5 files

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import numpy as np
from tables import Int16Col, Float32Col, StringCol
from tables import IsDescription


class pyeneHDF5Settings():
    '''pytables auxiliary'''
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

    def SaveSettings(self, fileh, EN, root):
        HDF5group = fileh.create_group(root, 'Core')
        HDF5table = fileh.create_table(HDF5group, "table",
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

        HDF5table = fileh.create_table(HDF5group, "Hydpopower_plants",
                                       self.PyeneHDF5Devices)
        HDF5row = HDF5table.row
        for xh in range(EN.NM.hydropower['Number']):
            HDF5row['location'] = EN.NM.hydropower['Bus'][xh]
            HDF5row['max'] = EN.NM.hydropower['Max'][xh]
            HDF5row['cost'] = EN.NM.hydropower['Cost'][xh]
            HDF5row['link'] = EN.NM.hydropower['Link'][xh]
            HDF5row.append()
        HDF5table.flush()

        HDF5table = fileh.create_table(HDF5group, "Pumps",
                                       self.PyeneHDF5Devices)
        HDF5row = HDF5table.row
        for xh in range(EN.NM.pumps['Number']):
            HDF5row['location'] = EN.NM.pumps['Bus'][xh]
            HDF5row['max'] = EN.NM.pumps['Max'][xh]
            HDF5row['cost'] = EN.NM.pumps['Value'][xh]
            HDF5row.append()
        HDF5table.flush()

        HDF5table = fileh.create_table(HDF5group, "RES_generators",
                                       self.PyeneHDF5Devices)
        HDF5row = HDF5table.row
        for xh in range(EN.NM.RES['Number']):
            HDF5row['location'] = EN.NM.RES['Bus'][xh]
            HDF5row['max'] = EN.NM.RES['Max'][xh]
            HDF5row['cost'] = EN.NM.RES['Cost'][xh]
            HDF5row['link'] = EN.NM.RES['Link'][xh]
            HDF5row.append()
        HDF5table.flush()

    def saveResults(self, fileh, EN, mod, root, SimNo):
        HDF5group = fileh.create_group(root, 'Simulation_{:05d}'.format(SimNo))
        HDF5aux = np.zeros((EN.NM.scenarios['NoDem'],
                            EN.NM.settings['NoTime']), dtype=float)
        xp = 0
        for xs in range(EN.NM.scenarios['NoDem']):
            for xt in range(EN.NM.settings['NoTime']):
                HDF5aux[xs][xt] = EN.NM.scenarios['Demand'][xp]
                xp += 1
        fileh.create_array(HDF5group, "Demand_profiles", HDF5aux)

        HDF5aux = np.zeros((EN.NM.scenarios['NoRES'],
                            EN.NM.settings['NoTime']), dtype=float)
        xp = 0
        for xs in range(EN.NM.scenarios['NoRES']):
            for xt in range(EN.NM.settings['NoTime']):
                HDF5aux[xs][xt] = EN.NM.scenarios['RES'][xp] * \
                    EN.NM.networkE.graph['baseMVA']
                xp += 1
        fileh.create_array(HDF5group, "RES_profiles", HDF5aux)

        # Hydropower allowance
        aux = np.zeros(EN.EM.settings['Vectors'], dtype=float)
        if type(mod.WInFull) is np.ndarray:
            if EN.EM.settings['Vectors'] == 1:
                aux[0] = mod.WInFull[1]
            else:
                for xv in mod.sVec:
                    aux[xv] = mod.WInFull[1][xv]
        else:
            for xv in mod.sVec:
                aux[xv] = mod.WInFull[1, xv].value

        fileh.create_array(HDF5group, "Hydro_Allowance", aux)

        hp_marginal = np.zeros(EN.EM.settings['Vectors'], dtype=float)
        for xi in range(EN.EM.settings['Vectors']):
            hp_marginal[xi] = EN.get_HydroMarginal(mod, xi+1)
        fileh.create_array(HDF5group, "Hydro_Marginal", hp_marginal)

        for xs in range(EN.NM.scenarios['Number']):
            HDF5table = fileh.create_table(HDF5group,
                                           "Scenario_{:02d}".format(xs),
                                           self.PyeneHDF5Results)
            HDF5row = HDF5table.row
            for xt in range(EN.NM.settings['NoTime']):
                HDF5row['time'] = xt
                HDF5row['generation'] = \
                    EN.get_AllGeneration(mod, 'Conv', 'snapshot', times=[xt],
                                         scens=[xs])
                HDF5row['hydropower'] = \
                    EN.get_AllGeneration(mod, 'Hydro', 'snapshot', times=[xt],
                                         scens=[xs])
                HDF5row['RES'] = EN.get_AllGeneration(mod, 'RES', 'snapshot',
                                                      times=[xt], scens=[xs])
                HDF5row['spill'] = EN.get_AllRES(mod, 'snapshot', times=[xt],
                                                 scens=[xs])
                HDF5row['demand'] = EN.get_AllDemand(mod, 'snapshot',
                                                     times=[xt], scens=[xs])
                HDF5row['pump'] = EN.get_AllPumps(mod, 'snapshot', times=[xt],
                                                  scens=[xs])
                HDF5row['loss'] = EN.get_AllLoss(mod, 'snapshot', times=[xt],
                                                 scens=[xs])
                HDF5row['curtailment'] = \
                    EN.get_AllDemandCurtailment(mod, 'snapshot', times=[xt],
                                                scens=[xs])
                HDF5row.append()
            HDF5table.flush()
