# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:23:14 2018

Pyene Devices collects the data, location and connections of the different
devices considered in pyene

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import math

'''                          CONFIGURATION CLASSES                          '''


class BusConfig:
    ''' Default settings for an electricity bus '''
    def __init__(self):
        # Basic settings
        self.settings = {
                'BASE_KV': None,
                'BS': None,
                'BUS_AREA': None,
                'BUS_TYPE': None,
                'BUS_X': None,  # Coordinates X
                'BUS_Y': None,  # Coordinates Y
                'Demand': [],  # Demand time series
                'GS': None,
                'Peak': None,  # Peak demand (MW)
                'Name': None,  # Bus name
                'Number': None,  # Bus number
                'VM': None,
                'VA': None,
                'VMAX': None,
                'VMIN': None,
                'ZONE': None
                }

    def MPCconfigure(self, mpc, No=0):
        ''' Configure using mat power data '''

        self.settings['BASE_KV'] = mpc['BASE_KV'][No]
        self.settings['BS'] = mpc['BS'][No]
        self.settings['BUS_AREA'] = mpc['BUS_AREA'][No]
        self.settings['BUS_TYPE'] = mpc['BUS_TYPE'][No]
        self.settings['BUS_X'] = mpc['BUS_X'][No]
        self.settings['BUS_Y'] = mpc['BUS_Y'][No]
        # TODO: Demand
        self.settings['GS'] = mpc['GS'][No]
        # TODO: Peak
        # TODO: Name
        self.settings['Number'] = No
        self.settings['VM'] = mpc['VM'][No]
        self.settings['VA'] = mpc['VA'][No]
        self.settings['VMAX'] = mpc['VMAX'][No]
        self.settings['VMIN'] = mpc['VMIN'][No]
        self.settings['ZONE'] = mpc['ZONE'][No]


class BranchConfig:
    ''' Electricity Branch '''
    def __init__(self):
        # Basic settings
        self.settings = {
                'ANGMAX': None,
                'ANGMIN': None,
                'BR_B': None,
                'BR_R': None,
                'BR_STATUS': None,
                'BR_X': None,
                'Number': None,  # Branch number
                'F_BUS': None,  # Bus (from)
                'RATE_A': None,
                'RATE_A': None,
                'RATE_C': None,
                'T_BUS': None  # Bus (to)
                }

    def MPCconfigure(self, mpc, No=0):
        ''' Configure using mat power data '''
        print()
        print(mpc)
        print()

        self.settings['ANGMAX'] = mpc['ANGMAX'][No]
        self.settings['ANGMIN'] = mpc['ANGMIN'][No]
        self.settings['BR_B'] = mpc['BR_B'][No]
        self.settings['BR_R'] = mpc['BR_R'][No]
        self.settings['BR_STATUS'] = mpc['BR_STATUS'][No]
        self.settings['BR_X'] = mpc['BR_X'][No]
        self.settings['Number'] = No
        self.settings['F_BUS'] = mpc['F_BUS'][No]
        self.settings['RATE_A'] = mpc['RATE_A'][No]
        self.settings['RATE_B'] = mpc['RATE_B'][No]
        self.settings['RATE_C'] = mpc['RATE_C'][No]
        self.settings['T_BUS'] = mpc['T_BUS'][No]
        
        

'''                               DEVICE CLASSES                            '''
class Bus:
    ''' Electricity bus '''
    def __init__(self, obj=None):
        ''' Initialise network class '''
        # Get default values
        if obj is None:
            obj = BusConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))


class ENet:
    ''' Electricity network '''
    def __init__(self, NoBus=1, NoBranch=1):
        ''' General electricity network settings '''
        self.settings = {
                'version': None,
                'baseMVA': None,
                'NoGen': None,
                'Slack': None,
                'Buses': None,  # Number of buses
                'Branches': None  # Number of buses
                }

        # Define bus objects - configuration class
        self.BusConfig = [BusConfig() for x in range(NoBus)]

        # Define branch objects - configuration class
        self.BranchConfig = [BranchConfig() for x in range(NoBranch)]

    def MPCconfigure(self, mpc):
        ''' Initialize using mat power data '''

        # General electricity network settings
        self.settings['Buses'] = mpc['NoBus']
        self.settings['Branches'] = mpc["NoBranch"]
        for xa in ['version', 'baseMVA', 'NoGen', 'Slack']:
            self.settings[xa] = mpc[xa]

#        # Bus data
#        for x in range(mpc['NoBus']):
#            self.BusConfig[x].MPCconfigure(mpc['bus'], x)
#
        # Branch data
        for x in range(mpc['NoBranch']):
            self.BranchConfig[x].MPCconfigure(mpc['branch'], x)

    

class ELineConfig:
    ''' Default settings for an electricity bus '''
    def __init__(self):
        # Basic settings
        self.settings = {
                'Number': None  # Bus number
                }

class ELine:
    ''' Electricity bus '''
    def __init__(self, obj=None):
        ''' Initialise network class '''
        # Get default values
        if obj is None:
            obj = BusConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))


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
        # Location of variables in pyomo
        self.no['vNGen'] = None
        self.no['vNGCost'] = None
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
