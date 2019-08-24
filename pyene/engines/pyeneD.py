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
        aux = ['BASE_KV', 'BS', 'BUS_AREA', 'BUS_TYPE', 'BUS_X', 'BUS_Y',
               'Demand', 'GS', 'PeakP', 'PeakQ', 'Position', 'Name', 'Number',
               'VM', 'VA', 'VMAX', 'VMIN', 'ZONE']
        self.settings = {}
        for x in aux:
            self.settings[x] = None

    def MPCconfigure(self, mpc, No=0):
        ''' Configure using mat power data '''

        self.settings['Position'] = No
        self.settings['Number'] = mpc['BUS_I'][No]
        self.settings['PeakP'] = mpc['PD'][No]
        self.settings['PeakP'] = mpc['QD'][No]

        aux = ['BASE_KV', 'BS', 'BUS_AREA', 'BUS_TYPE', 'GS', 'VM', 'VA',
               'VMAX', 'VMIN', 'ZONE']
        for x in aux:
            self.settings[x] = mpc[x][No]

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
        aux = ['ANGMAX', 'ANGMIN', 'BR_B', 'BR_R', 'BR_STATUS', 'BR_X',
               'Number', 'F_BUS', 'RATE_A', 'RATE_A', 'RATE_C', 'TAP', 'T_BUS']
        self.settings = {}
        for x in aux:
            self.settings[x] = None

    def MPCconfigure(self, mpc, No=0):
        ''' Configure using mat power data '''

        self.settings['Number'] = No+1
        self.settings['Position'] = No

        aux = ['ANGMAX', 'ANGMIN', 'BR_B', 'BR_R', 'BR_STATUS', 'BR_X',
               'F_BUS', 'RATE_A', 'RATE_B', 'RATE_C', 'TAP', 'T_BUS']
        for x in aux:
            self.settings[x] = mpc[x][No]


class ConventionalConfig:
    ''' Conventnional generator '''
    def __init__(self):
        # Basic settings
        aux = ['Ancillary', 'APF', 'GEN', 'GEN_BUS', 'MBASE', 'PC1', 'PC2',
               'PG', 'PMAX', 'PMIN', 'QC1MIN', 'QC1MAX', 'QC2MIN', 'QC2MAX',
               'QG', 'QMAX', 'QMIN', 'Ramp', 'RAMP_AGC', 'RAMP_10', 'RAMP_30',
               'RAMP_Q', 'RES', 'VG']
        self.settings = {}
        for x in aux:
            self.settings[x] = None

        # Cost data
        aux = ['COST', 'MODEL', 'NCOST', 'SHUTDOWN', 'STARTUP']
        self.cost = {}
        for x in aux:
            self.cost[x] = None

    def MPCconfigure(self, mpc, conv, No=0):
        ''' Configure using mat power data '''

        # Generator settings - from mat power file
        self.settings['Position'] = No
        aux = ['APF', 'GEN', 'GEN_BUS', 'MBASE', 'PC1', 'PC2', 'PG', 'PMAX',
               'PMIN', 'QC1MIN', 'QC1MAX', 'QC2MIN', 'QC2MAX', 'QG', 'QMAX',
               'QMIN', 'RAMP_AGC', 'RAMP_10', 'RAMP_30', 'RAMP_Q', 'VG']
        for x in aux:
            self.settings[x] = mpc['gen'][x][No]

        # Generator costs - from mat power file
        aux = ['COST', 'MODEL', 'NCOST', 'SHUTDOWN', 'STARTUP']
        for x in aux:
            self.cost[x] = mpc['gencost'][x][No]

        # Generator data - from configuration file
        aux = ['Ancillary', 'Ramp', 'RES']
        for x in aux:
            self.cost[x] = conv[x]


class HydropowerConfig:
    ''' Hydropower generator '''
    def __init__(self):
        # Basic settings
        aux = ['Ancillary', 'Baseload', 'Bus', 'Max', 'Ramp',
               'RES', 'Position']
        self.settings = {}
        for x in aux:
            self.settings[x] = None

        aux = ['MODEL', 'NCOST', 'COST']
        self.cost = {}
        for x in aux:
            self.cost[x] = None

    def MPCconfigure(self, hydro, No=0):
        ''' Configure using hydropower settings '''
        self.settings['Position'] = No
        aux = ['Ancillary', 'Baseload', 'Ramp', 'RES']
        for x in aux:
            self.settings[x] = hydro[x]

        self.settings['Bus'] = hydro['Bus'][No]
        self.settings['Max'] = hydro['Max'][No]

        # Default cost model
        self.cost['MODEL'] = 2
        self.cost['NCOST'] = 1
        self.cost['COST'] = hydro['Cost'][No]


class RESConfig:
    ''' RES generator '''
    def __init__(self):
        # Basic settings
        aux = ['Bus', 'Cost', 'Max', 'Uncertainty', 'Position']
        self.settings = {}
        for x in aux:
            self.settings[x] = None

        aux = ['MODEL', 'NCOST', 'COST']
        self.cost = {}
        for x in aux:
            self.cost[x] = None

    def MPCconfigure(self, RES, No=0):
        ''' Configure using hydropower settings '''
        self.settings['Position'] = No
        self.settings['Bus'] = RES['Bus'][No]
        self.settings['Max'] = RES['Max'][No]
        self.settings['Uncertainty'] = RES['Uncertainty']

        # Default cost model
        self.cost['MODEL'] = 2
        self.cost['NCOST'] = 1
        self.cost['COST'] = RES['Cost'][No]


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

    def cNEFlow_rule(self, m, xt, xs, ConF, ConV, Bus):
        ''' Set DC power flow constraint '''
        if self.is_active(xs):
            xaux1 = ConV+Bus[self.get_PosF()].get_Sec(xs)
            xaux2 = ConV+Bus[self.get_PosT()].get_Sec(xs)

            return m.vNFlow[ConF+self.get_Sec(xs), xt] == \
                (m.vNVolt[xaux1, xt]-m.vNVolt[xaux2, xt])/self.data['BR_X']
        else:
            return Constraint.Skip

    def cNEFMax_rule(self, m, xt, xs, ConF):
        ''' Branch capacity constraint (positive) '''
        if self.is_active(xs):
            return m.vNFlow[ConF+self.get_Sec(xs), xt] >= \
                -self.data['RATE_A']
        else:
            return Constraint.Skip

    def cNEFMin_rule(self, m, xt, xs, ConF):
        ''' Branch capacity constraint (positive) '''
        if self.is_active(xs):
            return m.vNFlow[ConF+self.get_Sec(xs), xt] <= \
                self.data['RATE_A']
        else:
            return Constraint.Skip

    def cNDCLossA_rule(self, m, xt, xL, ConF, ConL, A, B):
        ''' Power losses estimation - Positive '''
        return m.vNLoss[ConL+self.get_Pos(), xt] >= \
            (A[xL]+B[xL]*m.vNFlow[ConF+self.get_Pos(), xt]) * \
            self.data['BR_R']

    def cNDCLossB_rule(self, m, xt, xL, ConF, ConL, A, B):
        ''' Power losses estimation - Negative '''
        return m.vNLoss[ConL+self.get_Pos(), xt] >= \
            (A[xL]-B[xL]*m.vNFlow[ConF+self.get_Pos(), xt]) * \
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
        self.data['GenType'] = []  # Types of generators connected to the bus
        self.data['GenPosition'] = []  # Position of the generators

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
            self.Bus[xf].data['F_Branches'].append(ob.data['Position'])
            self.Bus[xt].data['T_Branches'].append(ob.data['Position'])
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
        for xb in self.Bus[xn].get_TBranch():  # Branches connected to the bus
            # Is teh branch active in the scenario?
            if self.Branch[xb].pyomo['N-1'][xs] is not None:
                aux.append(self.Branch[xb].pyomo['N-1'][xs])
        return aux

    def get_FFlow(self, xn, xs):
        ''' Get branches connected from bus per scenario '''
        aux = []
        for xb in self.Bus[xn].get_FBranch():  # Branches connected to the bus
            # Is teh branch active in the scenario?
            if self.Branch[xb].pyomo['N-1'][xs] is not None:
                aux.append(self.Branch[xb].pyomo['N-1'][xs])
        return aux
        aux = []

    def cNEFlow_rule(self, m, xt, xb, xs, ConF, ConV):
        ''' Branch flows constraint '''
        return self.Branch[xb].cNEFlow_rule(m, xt, xs, ConF, ConV, self.Bus)

    def cNEFMax_rule(self, m, xt, xb, xs, ConF):
        ''' Branch capacity constraint (positive) '''
        return self.Branch[xb].cNEFMax_rule(m, xt, xs, ConF)

    def cNEFMin_rule(self, m, xt, xb, xs, ConF):
        ''' Branch capacity constraint (positive) '''
        return self.Branch[xb].cNEFMin_rule(m, xt, xs, ConF)

    def cNDCLossA_rule(self, m, xt, xb, xL, ConF, ConL):
        ''' Power losses estimation - Positive '''
        return self.Branch[xb].cNDCLossA_rule(m, xt, xL, ConF, ConL,
                                              self.loss['A'], self.loss['B'])

    def cNDCLossB_rule(self, m, xt, xb, xL, ConF, ConL):
        ''' Power losses estimation - Negative '''
        return self.Branch[xb].cNDCLossB_rule(m, xt, xL, ConF, ConL,
                                              self.loss['A'], self.loss['B'])

    def findBusPosition(self, bus):
        ''' Find the position of a bus

        This is required as the bus numbers may not begin from zero, or some
        positions may be missing
        '''
        xn = 0
        while self.Bus[xn].data['Number'] != bus:
            xn += 1

        return xn


class Conventional:
    ''' Conventional generator '''
    def __init__(self, obj):
        ''' Initialise generator class

        The class can use the following parameters:
        ['APF', 'GEN', 'MBASE', 'PC1', 'PC2', 'PG', 'QC1MIN', 'QC1MAX',
        'QC2MIN', 'QC2MAX', 'QG', 'QMAX', 'QMIN', 'RAMP_AGC',
        'RAMP_10', 'RAMP_30', 'RAMP_Q', 'RES', 'VG']
        However, only the ones that are currently used are passed
        ['COST', 'MODEL', 'NCOST', 'SHUTDOWN', 'STARTUP']
        '''
        # Parameters currently in use
        aux = ['Ancillary', 'PMAX', 'PMIN', 'Ramp', 'Position']

        # Get settings
        self.data = {}
        for xa in aux:
            self.data[xa] = obj.settings[xa]
        self.data['Bus'] = obj.settings['GEN_BUS']

        aux = ['COST', 'MODEL', 'NCOST', 'SHUTDOWN', 'STARTUP']
        self.cost = {}
        for xa in aux:
            self.cost[xa] = obj.cost[xa]


class Hydropower:
    ''' Hydropower generator '''
    def __init__(self, obj):
        ''' Initialise hydropower generator class

        The class can use the following parameters:
        ['Ancillary', 'Baseload', 'Bus', 'Max', 'Ramp', 'RES', 'Position']
        ['COST', 'MODEL', 'NCOST', 'SHUTDOWN', 'STARTUP']
        However, only the ones that are currently used are passed
        '''
        # Parameters currently in use
        aux = ['Ancillary', 'Baseload', 'Bus', 'Max', 'Ramp', 'RES',
               'Position']

        # Get settings
        self.data = {}
        for xa in aux:
            self.data[xa] = obj.settings[xa]

        aux = ['COST', 'MODEL', 'NCOST']
        self.cost = {}
        for xa in aux:
            self.cost[xa] = obj.cost[xa]


class RES:
    ''' RES generation '''
    def __init__(self, obj):
        ''' Initialise hydropower generator class

        The class can use the following parameters:
        ['Bus', 'Cost', 'Max', 'Uncertainty', 'Position']
        ['MODEL', 'NCOST', 'COST']
        However, only the ones that are currently used are passed
        '''
        # Parameters currently in use
        aux = ['Bus', 'Cost', 'Max', 'Uncertainty', 'Position']

        # Get settings
        self.data = {}
        for xa in aux:
            self.data[xa] = obj.settings[xa]

        aux = ['MODEL', 'NCOST', 'COST']
        self.cost = {}
        for xa in aux:
            self.cost[xa] = obj.cost[xa]


class Generators:
    ''' Electricity generators '''
    def __init__(self, NoConv=0, NoHydro=0, NoRES=0):
        ''' General generator settings '''
        self.data = {
                'Conv': NoConv,
                'Hydro': NoHydro,
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
        for x in range(self.data['Conv']):
            self.ConvConf[x].MPCconfigure(mpc, conv, x)

        for x in range(self.data['Hydro']):
            self.HydroConf[x].MPCconfigure(hydro, x)

        for x in range(self.data['RES']):
            self.RESConf[x].MPCconfigure(RES, x)

    def initialise(self, ENetwork):
        ''' Prepare objects and remove configuration versions '''
        # Initialise conventional generation object
        self.Conv = [Conventional(self.ConvConf[x]) for x in
                     range(self.data['Conv'])]
        del self.ConvConf

        # Initialise hydropower generator objects
        self.Hydro = [Hydropower(self.HydroConf[x]) for x in
                      range(self.data['Hydro'])]
        del self.HydroConf

        # Initialize RES generator objects
        self.RES = [RES(self.RESConf[x]) for x in range(self.data['RES'])]
        del self.RESConf

        # Link generators and buses
        genaux = ['Conv', 'Hydro', 'RES']
        xt = 0
        for ax in genaux:
            for ob in getattr(self, ax):
                print(ob.data)
                xb = ENetwork.findBusPosition(ob.data['Bus'])  # Bus
                xp = ob.data['Position']  # Generator
                # The Generator knows the position of its bus
                ob.data['BusPosition'] = xb
                # The bus knows the type and location of the generator
                ENetwork.Bus[xb].data['GenType'].append(xt)
                ENetwork.Bus[xb].data['GenPosition'].append(xp)
            xt += 1
