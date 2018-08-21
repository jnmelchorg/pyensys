# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:23:14 2018

Pyene Devices collects the data, location and connections of the different
devices considered in pyene

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import math


class pyeneDConfig:
    ''' Default settings used for this class '''
    def __init__(self):
        self.name = None  # Name of the device

        # Connections to networks in other models
        self.node = {
                'EM': None,  # Connection to the energy model
                'NM': None,  # Connection to the electricity network
                'HM': None  # Connection to the hydraulic network
                }
        # Connections to specific variables
        self.no = {}
        # Specific data for the device
        self.data = {}
        # Flags fro printing data
        self.print_flag = {}


class pyeneGeneratorConfig:
    ''' Default settings used by generators '''
    def __init__(self):
        self.print_flag['gen'] = True
        self.gen = {
                'PG': None,
                'QG': None,
                'QMAX': None,
                'QMIN': None,
                'VG': None,
                'MBASE': None,
                'GEN': None,
                'PMAX': None,
                'PMIN': None,
                'PC1': None,
                'PC2': None,
                'QC1MIN': None,
                'QC1MAX': None,
                'QC2MIN': None,
                'QC2MAX': None,
                'RAMP_AGC': None,
                'RAMP_10': None,
                'RAMP_30': None,
                'RAMP_Q': None,
                'APF': None,
                'MODEL': None,
                'STARTUP': None,
                'SHUTDOWN': None,
                'NCOST': None,
                'COST': None
                }
#        # Location of variables in pyomo
#        self.no['vNGen'] = None
#        self.no['vNGCost'] = None
#        # Location of variables in a matrix
#        self.var['vNGen'] = None
#        self.var['vNGCost'] = None

    def print_gen(self):
        ''' Display generation settings '''
        print(self.node['NM'], 'GEN_Bus: bus number')
        print(self.gen['PG'], 'PG real power output (MW)')
        print(self.gen['QG'], 'QG reactive power output (MVAr)')
        print(self.gen['QMAX'], 'QMAX maximum reactive power output (MVAr)')
        print(self.gen['QMIN'], 'QMIN minimum reactive power output (MVAr)')
        print(self.gen['VG'], 'VG voltage magnitude setpoint (p.u.)')
        print(self.gen['MBASE'], 'MBASE total MVA base of machine, defaults' +
              'to baseMVA')
        print(self.gen['GEN'], 'GEN machine status (1=in-service,' +
              '0=out-of-service)')
        print(self.gen['PMAX'], 'PMAX maximum real power output (MW)')
        print(self.gen['PMIN'], 'PMIN minimum real power output (MW)')
        print(self.gen['PC1'], 'PC1 lower real power output of PQ capability' +
              'curve (MW)')
        print(self.gen['PC2'], 'PC2 upper real power output of PQ capability' +
              'curve (MW)')
        print(self.gen['QC1MIN'], 'QC1MIN minimum reactive power output' +
              'at PC1 (MVAr)')
        print(self.gen['QC1MAX'], 'QC1MAX maximum reactive power output at' +
              'PC1 (MVAr)')
        print(self.gen['QC2MIN'], 'QC2MIN minimum reactive power output at' +
              'PC2 (MVAr)')
        print(self.gen['QC2MAX'], 'QC2MAX maximum reactive power output at' +
              'PC2 (MVAr)')
        print(self.gen['RAMP_AGC'], 'RAMP_AGC ramp rate for load' +
              'following/AGC (MW/min)')
        print(self.gen['RAMP_10'], 'RAMP_10 ramp rate for 10 minute' +
              'reserves (MW)')
        print(self.gen['RAMP_30'], 'RAMP_30 ramp rate for 30 minute' +
              'reserves (MW)')
        print(self.gen['RAMP_Q'], 'RAMP_Q ramp rate for reactive power' +
              '(2 sec timescale) (MVAr/min)')


class DeviceClass:
    def get_Node(self, txt):
        ''' Return connection to a given node '''
        return self.node[txt]

    def check_Node(self, txt, Node):
        ''' Check if it is connected to a given node'''
        if Node == self.node[txt]:
            return self.node[txt]
        else:
            return 0

    def get_No(self, txt, Size=1, Scen=0):
        ''' Return position in a scenario '''
        return self.no[txt]+Size*Scen

    def check_No(self, txt, pos, Size=1):
        ''' Check if the position matches this device '''
        if pos % Size == self.no[txt]:
            return math.floor(pos % Size)
        else:
            return None

    def printDevice(self):
        print('\nDevice: ', self.name)
        print('Has the following settings')
        for (key, val) in self.data:
            print(key, val)
        print('Is connected to:')
        for (key, val) in self.node:
            print(key, val)
        print('As well as to variables:')
        for (key, val) in self.no:
            print(key, val)


class ConvClass(pyeneDConfig, pyeneGeneratorConfig, DeviceClass):
    ''' Conventionsl generator '''
    def __init__(self):
        ''' Get general and generator configurations '''
        pyeneDConfig.__init__(self)
        pyeneGeneratorConfig.__init__(self)



class RESClass(ConvClass):
    ''' RES generator '''
    def __init__(self):
        ''' Get general and generator configurations '''
        ConvClass.__init__(self)



class HydroClass(ConvClass):
    ''' COnventional generator '''
    def __init__(self):
        ''' Get general and generator configurations '''
        ConvClass.__init__(self)
