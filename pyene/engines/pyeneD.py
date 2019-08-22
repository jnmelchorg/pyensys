# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:23:14 2018

Pyene Devices collects the data, location and connections of the different
devices considered in pyene

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import math
import numpy as np
import copy
from pyomo.core import Constraint

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
                'Demand': None,  # Demand time series
                'GS': None,
                'PeakP': None,  # Peak demand (MW)
                'PeakQ': None,  # Peak demand (MVAr)
                'Position': None,  # Position of the data in mpc
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
        # TODO: Demand
        self.settings['GS'] = mpc['GS'][No]
        self.settings['PeakP'] = mpc['PD'][No]
        self.settings['PeakP'] = mpc['QD'][No]
        self.settings['Position'] = No
        # TODO: Name
        self.settings['Number'] = mpc['BUS_I'][No]
        self.settings['VM'] = mpc['VM'][No]
        self.settings['VA'] = mpc['VA'][No]
        self.settings['VMAX'] = mpc['VMAX'][No]
        self.settings['VMIN'] = mpc['VMIN'][No]
        self.settings['ZONE'] = mpc['ZONE'][No]

        #  Optional data - not included in all files
        if 'BUS_NAME' in mpc.keys():
            self.settings['Name'] = mpc['BUS_NAME'][No]
        if 'BUS_X' in mpc.keys():
            self.settings['BUS_X'] = mpc['BUS_X'][No]
            self.settings['BUS_Y'] = mpc['BUS_Y'][No]


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
                'TAP': None,
                'T_BUS': None  # Bus (to)
                }

    def MPCconfigure(self, mpc, No=0):
        ''' Configure using mat power data '''

        self.settings['ANGMAX'] = mpc['ANGMAX'][No]
        self.settings['ANGMIN'] = mpc['ANGMIN'][No]
        self.settings['BR_B'] = mpc['BR_B'][No]
        self.settings['BR_R'] = mpc['BR_R'][No]
        self.settings['BR_STATUS'] = mpc['BR_STATUS'][No]
        self.settings['BR_X'] = mpc['BR_X'][No]
        self.settings['Number'] = No+1
        self.settings['F_BUS'] = mpc['F_BUS'][No]
        self.settings['Position'] = No
        self.settings['RATE_A'] = mpc['RATE_A'][No]
        self.settings['RATE_B'] = mpc['RATE_B'][No]
        self.settings['RATE_C'] = mpc['RATE_C'][No]
        self.settings['TAP'] = mpc['TAP'][No]
        self.settings['T_BUS'] = mpc['T_BUS'][No]


class ConventionalConfig:
    ''' Conventnional generator '''
    def __init__(self):
        # Basic settings
        self.settings = {
                'Ancillary': None,  # Can it provide ancillary services?
                'APF': None,
                'GEN': None,
                'GEN_BUS': None,
                'MBASE': None,
                'PC1': None,
                'PC2': None,
                'PG': None,
                'PMAX': None,
                'PMIN': None,
                'QC1MIN': None,
                'QC1MAX': None,
                'QC2MIN': None,
                'QC2MAX': None,
                'QG': None,
                'QMAX': None,
                'QMIN': None,
                'Ramp': None,  # Set ramps for conventional generators
                'RAMP_AGC': None,
                'RAMP_10': None,
                'RAMP_30': None,
                'RAMP_Q': None,
                'RES': True,  # Can it support RES integration?
                'VG': None
                }

        # Cost data
        self.costs = {
                'COST': None,
                'MODEL': None,
                'NCOST': None,
                'SHUTDOWN': None,
                'STARTUP': None
                }

    def MPCconfigure(self, mpc, conv, No=0):
        ''' Configure using mat power data '''

        # Generator settings - from mat power file
        self.settings['APF'] = mpc['gen']['APF'][No]
        self.settings['GEN'] = mpc['gen']['GEN'][No]
        self.settings['GEN_BUS'] = mpc['gen']['GEN_BUS'][No]
        self.settings['MBASE'] = mpc['gen']['MBASE'][No]
        self.settings['PC1'] = mpc['gen']['PC1'][No]
        self.settings['PC2'] = mpc['gen']['PC2'][No]
        self.settings['PG'] = mpc['gen']['PG'][No]
        self.settings['PMAX'] = mpc['gen']['PMAX'][No]
        self.settings['PMIN'] = mpc['gen']['PMIN'][No]
        self.settings['QC1MIN'] = mpc['gen']['QC1MIN'][No]
        self.settings['QC1MAX'] = mpc['gen']['QC1MAX'][No]
        self.settings['QC2MIN'] = mpc['gen']['QC2MIN'][No]
        self.settings['QC2MAX'] = mpc['gen']['QC2MAX'][No]
        self.settings['QG'] = mpc['gen']['QG'][No]
        self.settings['QMAX'] = mpc['gen']['QMAX'][No]
        self.settings['QMIN'] = mpc['gen']['QMIN'][No]
        self.settings['RAMP_AGC'] = mpc['gen']['RAMP_AGC'][No]
        self.settings['RAMP_10'] = mpc['gen']['RAMP_10'][No]
        self.settings['RAMP_30'] = mpc['gen']['RAMP_30'][No]
        self.settings['RAMP_Q'] = mpc['gen']['RAMP_Q'][No]
        self.settings['VG'] = mpc['gen']['VG'][No]

        # Generator costs - from mat power file
        self.costs['COST'] = mpc['gencost']['COST'][No]
        self.costs['MODEL'] = mpc['gencost']['MODEL'][No]
        self.costs['NCOST'] = mpc['gencost']['NCOST'][No]
        self.costs['SHUTDOWN'] = mpc['gencost']['SHUTDOWN'][No]
        self.costs['STARTUP'] = mpc['gencost']['STARTUP'][No]

        # Generator data - from configuration file
        self.settings['Ancillary'] = conv['Ancillary']
        self.settings['Ramp'] = conv['Ramp']
        self.settings['RES'] = conv['RES']


class HydropowerConfig:
    ''' Hydropower generator '''
    def __init__(self):
        # Basic settings
        self.settings = {
                'Ancillary': None,  # Can it provide ancillary services?
                'Baseload': None,  # 0-1 for the use of water for baseload
                'Bus': None,  # Location (Bus) in the network
                'Cost': None,  # Costs
                'Link': None,  # Position of hydropower plants
                'Max': None,  # Capacity (MW)
                'Ramp': None,  # Ramp
                'RES': None  # Can it support RES integration?
            }

    def MPCconfigure(self, hydro, No=0):
        ''' Configure using hydropower settings '''
        self.settings['Ancillary'] = hydro['Ancillary']
        self.settings['Baseload'] = hydro['Baseload']
        self.settings['Bus'] = hydro['Bus'][No]
        self.settings['Cost'] = hydro['Cost'][No]
        self.settings['Link'] = hydro['Link']
        self.settings['Max'] = hydro['Max'][No]
        self.settings['Ramp'] = hydro['Ramp']
        self.settings['RES'] = hydro['RES']


class RESConfig:
    ''' RES generator '''
    def __init__(self):
        # Basic settings
        self.settings = {
                'Bus': None,  # Location (Bus) in the network
                'Cost': None,  # Costs
                'Link': None,  # Position of hydropower plants
                'Max': None,  # Capacity (MW)
                'Uncertainty': None  # Introduce reserve needs
            }

    def MPCconfigure(self, RES, No=0):
        ''' Configure using hydropower settings '''
        self.settings['Bus'] = RES['Bus'][No]
        self.settings['Cost'] = RES['Cost'][No]
        self.settings['Link'] = RES['Link']
        self.settings['Max'] = RES['Max'][No]
        self.settings['Uncertainty'] = RES['Uncertainty']


'''                               DEVICE CLASSES                            '''


class Branch:
    ''' Electricity branch '''
    def __init__(self, obj):
        ''' Initialise bus class

        The class can use the following parameters:
        ['ANGMAX', 'ANGMIN', 'BR_STATUS', 'Number',
        'F_BUS', 'Position', 'RATE_A', 'RATE_B', 'RATE_C', 'T_BUS']
        However, only the ones that are currently used are passed
        '''

        aux = ['BR_R', 'BR_X', 'F_BUS', 'Position', 'RATE_A', 'T_BUS', 'TAP',
               'BR_B']

        # Get settings
        self.data = {}
        for xa in aux:
            self.data[xa] = obj.settings[xa]

        self.data['F_Position'] = None
        self.data['T_Position'] = None

        # Paremeters used for pyomo
        self.pyomo = {}
        self.pyomo['N-1'] = None

    def get_BusF(self):
        ''' Get bus number at beginning (from) of the branch '''
        return self.data['F_BUS']

    def get_BusT(self):
        ''' Get bus number at end (to) of the branch '''
        return self.data['T_BUS']

    def get_Number(self):
        ''' Get branch number - starting from one'''
        return self.data['Position']+1

    def get_Pos(self):
        ''' Get position of the branch - starting from zero'''
        return self.data['Position']

    def get_PosF(self):
        ''' Get bus position at beginning (from) of the branch '''
        return self.data['F_Position']

    def get_PosT(self):
        ''' Get bus position at end (to) of the branch '''
        return self.data['T_Position']

    def get_Sec(self, xs):
        ''' Get position in N-1 scenario '''
        return self.pyomo['N-1'][xs]

    def is_active(self, xs):
        ''' Is the line connected in this scenario? '''
        return self.pyomo['N-1'][xs] is not None

    def cNEFlow_rule(self, m, xt, xs, ConF, ConV, Bus, xshift):
        ''' Set DC power flow constraint '''
        if self.is_active(xs):
            xaux1 = ConV+Bus[self.get_PosF()].get_Sec(xs)
            xaux2 = ConV+Bus[self.get_PosT()].get_Sec(xs)

            return m.vNFlow[ConF+self.get_Sec(xs)+xshift, xt] == \
                (m.vNVolt[xaux1, xt]-m.vNVolt[xaux2, xt])/self.data['BR_X']
        else:
            return Constraint.Skip

    def cNEFMax_rule(self, m, xt, xs, ConF, xshift):
        ''' Branch capacity constraint (positive) '''
        if self.is_active(xs):
            return m.vNFlow[ConF+self.get_Sec(xs)+xshift, xt] >= \
                -self.data['RATE_A']
        else:
            return Constraint.Skip

    def cNEFMin_rule(self, m, xt, xs, ConF, xshift):
        ''' Branch capacity constraint (positive) '''
        if self.is_active(xs):
            return m.vNFlow[ConF+self.get_Sec(xs)+xshift, xt] <= \
                self.data['RATE_A']
        else:
            return Constraint.Skip

    def cNDCLossA_rule(self, m, xt, xL, ConF, ConL, A, B, xshift):
        ''' Power losses estimation - Positive '''
        return m.vNLoss[ConL+self.get_Pos()+xshift, xt] >= \
            (A[xL]+B[xL]*m.vNFlow[ConF+self.get_Pos()+xshift, xt]) * \
            self.data['BR_R']

    def cNDCLossB_rule(self, m, xt, xL, ConF, ConL, A, B, xshift):
        ''' Power losses estimation - Negative '''
        return m.vNLoss[ConL+self.get_Pos()+xshift, xt] >= \
            (A[xL]-B[xL]*m.vNFlow[ConF+self.get_Pos()+xshift, xt]) * \
            self.data['BR_R']

class Bus:
    ''' Electricity bus '''
    def __init__(self, obj):
        ''' Initialise bus class

        The class can use the following parameters:
        [, 'BS', 'BUS_AREA', 'BUS_X','BUS_Y','Demand',
        'GS', 'PeakP', 'PeakQ', 'Position', 'Name', 'Number', 'VA',
        , 'ZONE']
        However, only the ones that are currently used are passed
        '''
        # Parameters currently in use
        aux = ['BUS_X', 'BUS_Y', 'Demand', 'PeakP', 'PeakQ', 'Position',
               'Name', 'Number', 'BUS_TYPE', 'BASE_KV', 'VMAX', 'VMIN', 'VM']

        # Get settings
        self.data = {}
        for xa in aux:
            self.data[xa] = obj.settings[xa]

        # New data
        self.data['F_Branches'] = []  # Branches connected from the bus
        self.data['T_Branches'] = []  # Branches connected to the bus
        self.data['F_Loss'] = []  # Branches connected from the bus - Losses
        self.data['T_Loss'] = []  # Branches connected to the bus - Losses
        self.data['NoFB'] = 0  # Number of branches connected from the bus
        self.data['NoTB'] = 0  # Number of branches connected to the bus

        self.pyomo = {}
        self.pyomo['N-1'] = None

    def get_Number(self):
        ''' Get Bus number '''
        return self.data['Number']

    def get_Position(self):
        ''' Get Bus position - beginning from zero '''
        return self.data['Position']

    def get_FBranch(self):
        ''' Get list of branches connected to the bus in an N-1 scenario'''
        return self.data['F_Branches']

    def get_TBranch(self):
        ''' Get list of branches connected to the bus in an N-1 scenario'''
        return self.data['T_Branches']

    def get_FLoss(self):
        ''' Get list of branches connected to the bus in an N-1 scenario'''
        return self.data['F_Loss']

    def get_TLoss(self):
        ''' Get list of branches connected to the bus in an N-1 scenario'''
        return self.data['T_Loss']

    def get_Sec(self, xs):
        ''' Get position of variable in N-1 scenario '''
        return self.pyomo['N-1'][xs]

class ElectricityNetwork:
    ''' Electricity network '''
    def __init__(self, NoBus=1, NoBranch=1):
        ''' General electricity network settings '''
        self.data = {
                'version': None,
                'baseMVA': None,
                'NoGen': None,
                'Slack': None,
                'Buses': NoBus,  # Number of buses
                'Branches': NoBranch,  # Number of branches
                'Security': None,  # N-1 cases to consider
                'SecurityNo': None  # Number of N-1 cases
                }
        self.loss = {}
        self.loss['A'] = None
        self.loss['B'] = None

        # Define bus objects - configuration class
        self.BusConfig = [BusConfig() for x in range(NoBus)]

        # Define branch objects - configuration class
        self.BranchConfig = [BranchConfig() for x in range(NoBranch)]

    def MPCconfigure(self, mpc):
        ''' Initialize using mat power data '''

        # General electricity network settings
        for xa in ['version', 'baseMVA', 'NoGen', 'Slack']:
            self.data[xa] = mpc[xa]

        # Bus data
        for x in range(mpc['NoBus']):
            self.BusConfig[x].MPCconfigure(mpc['bus'], x)

        # Branch data
        for x in range(mpc['NoBranch']):
            self.BranchConfig[x].MPCconfigure(mpc['branch'], x)

    def initialise(self, sett):
        ''' Prepare objects and remove configuration versions '''
        # TODO: to be removed
        xshift = 1  # Beginning the vector with 0 or 1
        self.data['shift'] = xshift

        # Initialise bus object
        self.Bus = [Bus(self.BusConfig[x]) for x in
                    range(self.data['Buses'])]
        del self.BusConfig

        # Initialise branch object
        self.Branch = [Branch(self.BranchConfig[x]) for x in
                       range(self.data['Branches'])]
        del self.BranchConfig

        # Security constraints
        if sett['SecurityFlag']:  # Consider all N-1 constraints
            self.data['SecurityNo'] = self.data['Branches']
            self.data['N-1'] = range(self.data['Branches'])
        else:
            self.data['SecurityNo'] = len(sett['Security'])
            self.data['N-1'] = sett['Security']

        # Match buses and nodes
        xcou = 0
        for ob in self.Branch:
            ob.data['shift'] = xshift
            # Find position of the bus (may be different from the number)
            xf = self.findBusPosition(ob.data['F_BUS'])
            xt = self.findBusPosition(ob.data['T_BUS'])

            # The branch now includes the position of the buses
            ob.data['F_Position'] = xf
            ob.data['T_Position'] = xt

            # Enable branch for the first N-1 scenario (intact network)
            ob.pyomo['N-1'] = [None] * (self.data['SecurityNo']+1)
            ob.pyomo['N-1'][0] = xcou
            xcou += 1

            # Tbe bus now includes the position of the relevant branches
            self.Bus[xf].data['F_Branches'].append(ob.data['Position']+xshift)
            self.Bus[xt].data['T_Branches'].append(ob.data['Position']+xshift)
            self.Bus[xf].data['NoFB'] += 1
            self.Bus[xt].data['NoTB'] += 1

            # Adjust line capacity
            ob.data['RATE_A'] = ob.data['RATE_A']/self.data['baseMVA']

        # Initialize security data for nodes
        for ob in self.Bus:
            ob.pyomo['N-1'] = [None] * (self.data['SecurityNo']+1)
            ob.pyomo['N-1'][0] = ob.data['Position']

        # Enable branches in other scenarios (pyomo)
        xsec = 0
        for xs in self.data['N-1']:
            xsec += 1
            # Add N-1 information to buses
            for ob in self.Bus:
                ob.pyomo['N-1'][xsec] = \
                    xsec*self.data['Buses']+ob.data['Position']
            # Add N-1 information to branches
            for ob in (self.Branch[xb] for xb in range(self.data['Branches'])
                       if xb+1 != xs):
                ob.pyomo['N-1'][xsec] = xcou
                xcou += 1

        # Model losses
        if sett['Losses']:
            for ob in self.Bus:
                ob.data['F_Loss'] = ob.data['F_Branches']
                ob.data['T_Loss'] = ob.data['T_Branches']

    def get_Security(self, No):
        ''' Define time series to model security constraints '''
        return (x for x in range(self.data['Branches']) if x != No)

    def get_TFlow(self, xn, xs):
        ''' Get branches connected to bus per scenario '''
        aux = []
        for xb1 in self.Bus[xn].get_TBranch():  # Branches connected to the bus
            xb = xb1 - self.data['shift']
            # Is teh branch active in the scenario?
            if self.Branch[xb].pyomo['N-1'][xs] is not None:
                aux.append(self.Branch[xb].pyomo['N-1'][xs]+self.data['shift'])
        return aux

    def get_FFlow(self, xn, xs):
        ''' Get branches connected from bus per scenario '''
        aux = []
        for xb1 in self.Bus[xn].get_FBranch():  # Branches connected to the bus
            xb = xb1 - self.data['shift']
            # Is teh branch active in the scenario?
            if self.Branch[xb].pyomo['N-1'][xs] is not None:
                aux.append(self.Branch[xb].pyomo['N-1'][xs]+self.data['shift'])
        return aux
        aux = []

    def cNEFlow_rule(self, m, xt, xb, xs, ConF, ConV):
        ''' Branch flows constraint '''
        return self.Branch[xb].cNEFlow_rule(m, xt, xs, ConF, ConV, self.Bus, self.data['shift'])

    def cNEFMax_rule(self, m, xt, xb, xs, ConF):
        ''' Branch capacity constraint (positive) '''
        return self.Branch[xb].cNEFMax_rule(m, xt, xs, ConF, self.data['shift'])

    def cNEFMin_rule(self, m, xt, xb, xs, ConF):
        ''' Branch capacity constraint (positive) '''
        return self.Branch[xb].cNEFMin_rule(m, xt, xs, ConF, self.data['shift'])

    def cNDCLossA_rule(self, m, xt, xb, xL, ConF, ConL):
        ''' Power losses estimation - Positive '''
        return self.Branch[xb].cNDCLossA_rule(m, xt, xL, ConF, ConL,
                                              self.loss['A'], self.loss['B'],
                                              self.data['shift'])

    def cNDCLossB_rule(self, m, xt, xb, xL, ConF, ConL):
        ''' Power losses estimation - Negative '''
        return self.Branch[xb].cNDCLossB_rule(m, xt, xL, ConF, ConL,
                                              self.loss['A'], self.loss['B'],
                                              self.data['shift'])

    def findBusPosition(self, bus):
        ''' Find the position of a bus

        This is required as the bus numbers may not begin from zero, or some
        positions may be missing
        '''
        xn = 0
        while self.Bus[xn].data['Number'] != bus:
            xn += 1

        return xn


class Generators:
    ''' Electricity generators '''
    def __init__(self, NoConv=0, NoHydro=0, NoRES=0):
        ''' General generator settings '''
        self.settings = {
                'Conventional': NoConv,
                'Hydropower': NoHydro,
                'RES': NoRES
                }

        # Conventional generators
        self.ConvConf = [ConventionalConfig() for x in range(NoConv)]

        # Hydropower generators
        self.HydroConf = [HydropowerConfig() for x in range(NoHydro)]

        # RES generators
        self.RESConf = [RESConfig() for x in range(NoRES)]

    def MPCconfigure(self, mpc, conv, hydro, RES):
        ''' Initialize using mat power data '''

        # Conventional generators
        for x in range(self.settings['Conventional']):
            self.ConvConf[x].MPCconfigure(mpc, conv, x)

        for x in range(self.settings['Hydropower']):
            self.HydroConf[x].MPCconfigure(hydro, x)

        for x in range(self.settings['RES']):
            self.RESConf[x].MPCconfigure(RES, x)


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
