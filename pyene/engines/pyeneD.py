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
        self.position = {}
        # Specific data for the device
        self.data = {}


class pyeneDHydroConfig(pyeneDConfig):
    ''' Default settings used for this class '''
    def __init__(self):
        # Connections to other models
        pyeneDConfig.__init__()
#        super(pyeneDConfig, self).__init__()

        # Connection  to variables
        self.no['vNGen'] = None  # Connection to VNGen

        self.data['Max'] = None  # Capacity (kW)
        self.data['Cost'] = None  # Cost (OF)
        self.data['RES'] = None  # Location of the RES profiles


class pyeneDRESConfig(pyeneDConfig):
    ''' Default settings used for this class '''
    def __init__(self):
        # Connections to other models
        pyeneDConfig.__init__()
#        super(pyeneDConfig, self).__init__()

        # Connection  to variables
        self.no['vNGen'] = None  # Connection to VNGen

        self.data['Max'] = None  # Capacity (kW)
        self.data['Cost'] = None  # Cost (OF)
        self.data['RES'] = None  # Location of the RES profiles


class DeviceClass:
    def __init__(self, obj=None):
        ''' Initialise device class '''
        # Get default values
        if obj is None:
            obj = pyeneDConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

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


class RESClass(DeviceClass):
    ''' COnventional generator '''


class HydroClass(DeviceClass):
    ''' COnventional generator '''
