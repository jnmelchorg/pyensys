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
               'VM', 'VA', 'VMAX', 'VMIN', 'ZONE', 'Load_Type']
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
            if mpc['BUS_NAME'] != []:
                self.settings['Name'] = mpc['BUS_NAME'][No]
        if 'BUS_X' in mpc.keys():
            self.settings['BUS_X'] = mpc['BUS_X'][No]
            self.settings['BUS_Y'] = mpc['BUS_Y'][No]
        if 'Load_Type' in mpc.keys():
            self.settings['Load_Type'] = mpc['Load_Type'][No]
        if 'Loss_Fix' in mpc.keys():
            self.settings['Loss_Fix'] = mpc['Loss_Fix'][No]
        else:
            self.settings['Loss_Fix'] = 0


class BranchConfig:
    ''' Electricity Branch '''
    def __init__(self):
        # Basic settings
        aux = ['ANGMAX', 'ANGMIN', 'BR_B', 'BR_R', 'BR_STATUS', 'BR_X',
               'Number', 'F_BUS', 'RATE_A', 'RATE_A', 'RATE_C', 'TAP',
               'T_BUS', 'Loss_Fix']
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

        if 'Loss_Fix' in mpc.keys():
            self.settings['Loss_Fix'] = mpc['Loss_Fix'][No]
        else:
            self.settings['Loss_Fix'] = 0


class ConventionalConfig:
    ''' Conventnional generator '''
    def __init__(self):
        # Basic settings
        aux = ['Ancillary', 'APF', 'GEN_BUS', 'MBASE', 'PC1', 'PC2',
               'PG', 'PMAX', 'PMIN', 'QC1MIN', 'QC1MAX', 'QC2MIN', 'QC2MAX',
               'QG', 'QMAX', 'QMIN', 'Ramp', 'RAMP_AGC', 'RAMP_10', 'RAMP_30',
               'RAMP_Q', 'RES', 'VG', 'MDT', 'MUT']
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
        aux = ['APF', 'GEN_BUS', 'MBASE', 'PC1', 'PC2', 'PG', 'PMAX',
               'PMIN', 'QC1MIN', 'QC1MAX', 'QC2MIN', 'QC2MAX', 'QG', 'QMAX',
               'QMIN', 'RAMP_AGC', 'RAMP_10', 'RAMP_30', 'RAMP_Q', 'VG']
        for x in aux:
            self.settings[x] = mpc['gen'][x][No]

        # Generator costs - from mat power file
        aux = ['COST', 'MODEL', 'NCOST', 'SHUTDOWN', 'STARTUP']
        for x in aux:
            self.cost[x] = mpc['gencost'][x][No]

        # Ramp constraints
        aux = len(conv['Ramp'])
        if aux == 0:  # No values from settings
            if 'Ramp' in mpc['gen'].keys():
                # Values can be taken from settings
                self.settings['Ramp'] = mpc['gen']['Ramp'][No]
            else:
                self.settings['Ramp'] = None
        elif aux == 1:  # Assign same ramp to all generators
            self.settings['Ramp'] = conv['Ramp'][0]
        else:  # Assign a ramp to each generator
            if No <= aux:
                self.settings['Ramp'] = conv['Ramp'][No]
            else:  # But the information was not provided
                self.settings['Ramp'] = None

        # MUT and MDT constraints
        aux = ['MUT', 'MDT']
        for x in aux:
            auxL = len(conv[x])
            if auxL == 0:
                self.settings[x] = None
            elif auxL == 1:
                self.settings[x] = conv[x][0]
            else:
                self.settings[x] = conv[x][No]

        # Provision of services
        aux = ['Ancillary', 'Baseload', 'RES']
        for x in aux:
            if len(conv[x]) == 1:
                self.settings[x] = conv[x][0]
            else:
                self.settings[x] = conv[x][No]


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
            if len(hydro[x]) == 1:
                self.settings[x] = hydro[x][0]
            else:
                self.settings[x] = hydro[x][No]

        self.settings['Bus'] = hydro['Bus'][No]
        self.settings['Max'] = hydro['Max'][No]

        # Default cost model
        self.cost['MODEL'] = 1
        self.cost['NCOST'] = 2
        self.cost['COST'] = [0, 0, 1, hydro['Cost'][No]]


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
        ''' Configure using RES settings '''
        self.settings['Position'] = No
        self.settings['Bus'] = RES['Bus'][No]
        self.settings['Max'] = RES['Max'][No]
        self.settings['Uncertainty'] = RES['Uncertainty']

        # Default cost model
        self.cost['MODEL'] = 1
        self.cost['NCOST'] = 2
        self.cost['COST'] = [0, 0, 1, RES['Cost'][No]]


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
               'BR_B', 'Loss_Fix']

        # Get settings
        self.data = {}
        for xa in aux:
            self.data[xa] = obj.settings[xa]

        self.data['F_Position'] = None
        self.data['T_Position'] = None

        # Paremeters used for pyomo
        self.pyomo = {}
        self.pyomo['N-1'] = None

    def cNDCLossA_rule(self, m, xt, xL, ConF, ConL, A, B, Bse):
        ''' Power losses estimation - Positive '''
        return m.vNLoss[ConL+self.get_Pos(), xt] >= \
            self.data['Loss_Fix']/Bse + \
            (A[xL]+B[xL]*m.vNFlow[ConF+self.get_Pos(), xt]) * \
            self.data['BR_R']

    def cNDCLossB_rule(self, m, xt, xL, ConF, ConL, A, B, Bse):
        ''' Power losses estimation - Negative '''
        return m.vNLoss[ConL+self.get_Pos(), xt] >= \
            self.data['Loss_Fix']/Bse + \
            (A[xL]-B[xL]*m.vNFlow[ConF+self.get_Pos(), xt]) * \
            self.data['BR_R']

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

    def get_B(self):
        ''' Get Susceptance '''
        return self.data['BR_B']

    def get_BusF(self):
        ''' Get bus number at beginning (from) of the branch '''
        return self.data['F_BUS']

    def get_BusT(self):
        ''' Get bus number at end (to) of the branch '''
        return self.data['T_BUS']

    def getLoss(self):
        ''' Return non technical losses in the bus '''
        return self.data['Loss_Fix']

    def get_N1(self, x=':'):
        ''' Get values for a single N-1 condition '''
        return self.pyomo['N-1'][x]

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

    def get_R(self):
        ''' Get Resistance '''
        return self.data['BR_R']

    def get_Rate(self):
        ''' Get Rate A for normal operation conditions'''
        return self.data['RATE_A']

    def get_Sec(self, xs=':'):
        ''' Get position in N-1 scenario '''
        return self.pyomo['N-1'][xs]

    def get_Tap(self):
        ''' Get tap position '''
        return self.data['TAP']

    def get_X(self):
        ''' Get Reactance '''
        return self.data['BR_X']

    def is_active(self, xs):
        ''' Is the line connected in this scenario? '''
        return self.pyomo['N-1'][xs] is not None

    def set_B(self, val):
        ''' Set Susceptance'''
        self.data['BR_B'] = val

    def set_PosF(self, val):
        ''' Set bus position at beginning (from) of the branch '''
        self.data['F_Position'] = val

    def set_PosT(self, val):
        ''' Set bus position at end (to) of the branch '''
        self.data['T_Position'] = val

    def set_N1(self, val, x=None):
        ''' Set values for all conditions '''
        if x is None:
            self.pyomo['N-1'] = val
        else:
            self.pyomo['N-1'][x] = val

    def set_R(self, val):
        ''' Set Resistance'''
        self.data['BR_R'] = val

    def set_Rate(self, val):
        ''' Set Rate A for normal operation conditions'''
        self.data['RATE_A'] = val

    def set_X(self, val):
        ''' Set Reactance'''
        self.data['BR_X'] = val


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
               'Name', 'Number', 'BUS_TYPE', 'BASE_KV', 'VMAX', 'VMIN', 'VM',
               'Load_Type', 'Loss_Fix']

        # Get settings
        self.data = {}
        for xa in aux:
            self.data[xa] = obj.settings[xa]

        # New data
        if self.data['Load_Type'] is None:
            self.data['Load_Type'] = 0  # 0) Urban 1) Rural
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

    def add_BraF(self, val):
        ''' Append value to F_Branches - Branches connected from node'''
        self.data['F_Branches'].append(val)
        self.data['NoFB'] += 1

    def add_BraT(self, val):
        ''' Append value to T_Branches - Branches connected to node'''
        self.data['T_Branches'].append(val)
        self.data['NoTB'] += 1

    def add_Gen(self, xt, xp):
        ''' Append generator type and position '''
        self.data['GenType'].append(xt)
        self.data['GenPosition'].append(xp)

    def get_BraF(self):
        ''' Get list of branches connected from the bus in an N-1 scenario'''
        return self.data['F_Branches']

    def get_BraT(self):
        ''' Get list of branches connected to the bus in an N-1 scenario'''
        return self.data['T_Branches']

    def get_kV(self):
        ''' Get base kV '''
        return self.data['BASE_KV']

    def get_LossF(self):
        ''' Get list of branches connected from the bus - Losses'''
        return self.data['F_Loss']

    def get_LossT(self):
        ''' Get list of branches connected to the bus - Losses'''
        return self.data['T_Loss']

    def getLoss(self):
        ''' Return non technical losses in the bus '''
        return self.data['Loss_Fix']

    def get_LT(self):
        ''' Get load type (1:Urban, 2:Rural) '''
        return self.data['Load_Type']

    def get_Number(self):
        ''' Get Bus number '''
        return self.data['Number']

    def get_Pos(self):
        ''' Get Bus position - beginning from zero '''
        return self.data['Position']

    def get_Sec(self, xs=':'):
        ''' Get position of variable in N-1 scenario '''
        return self.pyomo['N-1'][xs]

    def get_Type(self):
        ''' Get bus type '''
        return self.data['BUS_TYPE']

    def get_VM(self):
        ''' Get Voltege magnitude (pu) '''
        return self.data['VM']

    def get_Vmax(self):
        ''' Get max voltage limit (pu) '''
        return self.data['VMAX']

    def get_Vmin(self):
        ''' Get max voltage limit (pu) '''
        return self.data['VMIN']

    def get_X(self):
        ''' Get X coordinates '''
        return self.data['BUS_X']

    def get_Y(self):
        ''' Get Y coordinates '''
        return self.data['BUS_Y']

    def set_LossF(self, val):
        ''' Set list of branches connected from the bus - Losses'''
        self.data['F_Loss'] = val

    def set_LossT(self, val):
        ''' Set list of branches connected to the bus - Losses'''
        self.data['T_Loss'] = val

    def set_N1(self, val, x=None):
        ''' Set values for all conditions '''
        if x is None:
            self.pyomo['N-1'] = val
        else:
            self.pyomo['N-1'][x] = val

    def set_LT(self, val):
        ''' Set load type (0:Urban, 1:Rural) '''
        self.data['Load_Type'] = val


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

    def cNDCLossA_rule(self, m, xt, xb, xL, ConF, ConL):
        ''' Power losses estimation - Positive '''
        return self.Branch[xb].cNDCLossA_rule(m, xt, xL, ConF, ConL,
                                              self.loss['A'], self.loss['B'],
                                              self.get_Base())

    def cNDCLossB_rule(self, m, xt, xb, xL, ConF, ConL):
        ''' Power losses estimation - Negative '''
        return self.Branch[xb].cNDCLossB_rule(m, xt, xL, ConF, ConL,
                                              self.loss['A'], self.loss['B'],
                                              self.get_Base())

    def cNEFlow_rule(self, m, xt, xb, xs, ConF, ConV):
        ''' Branch flows constraint '''
        return self.Branch[xb].cNEFlow_rule(m, xt, xs, ConF, ConV, self.Bus)

    def cNEFMax_rule(self, m, xt, xb, xs, ConF):
        ''' Branch capacity constraint (positive) '''
        return self.Branch[xb].cNEFMax_rule(m, xt, xs, ConF)

    def cNEFMin_rule(self, m, xt, xb, xs, ConF):
        ''' Branch capacity constraint (positive) '''
        return self.Branch[xb].cNEFMin_rule(m, xt, xs, ConF)

    def findBusPosition(self, bus):
        ''' Find the position of a bus

        This is required as the bus numbers may not begin from zero, or some
        positions may be missing
        '''
        xn = 0
        while self.Bus[xn].data['Number'] != bus:
            xn += 1

        return xn

    def get_Base(self):
        ''' Provide base MVA rating '''
        return self.data['baseMVA']

    def get_FlowF(self, xn, xs):
        ''' Get branches connected from bus per scenario '''
        aux = []
        for xb in self.Bus[xn].get_BraF():  # Branches connected to the bus
            # Is teh branch active in the scenario?
            if self.Branch[xb].get_N1(xs) is not None:
                aux.append(self.Branch[xb].get_N1(xs))
        return aux
        aux = []

    def get_FlowT(self, xn, xs):
        ''' Get branches connected to bus per scenario '''
        aux = []
        for xb in self.Bus[xn].get_BraT():  # Branches connected to the bus
            # Is teh branch active in the scenario?
            if self.Branch[xb].get_N1(xs) is not None:
                aux.append(self.Branch[xb].get_N1(xs))
        return aux

    def get_NoBra(self):
        ''' Get total number of branches in the network '''
        return self.data['Branches']

    def get_NoBus(self):
        ''' Get total number of buses in the network '''
        return self.data['Buses']

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
            xf = self.findBusPosition(ob.get_BusF())
            xt = self.findBusPosition(ob.get_BusT())

            # The branch now includes the position of the buses
            ob.set_PosF(xf)
            ob.set_PosT(xt)

            # Enable branch for the first N-1 scenario (intact network)
            ob.set_N1([None]*(self.data['SecurityNo']+1))
            ob.set_N1(xcou, 0)
            xcou += 1

            # Tbe bus now includes the position of the relevant branches
            self.Bus[xf].add_BraF(ob.get_Pos())
            self.Bus[xt].add_BraT(ob.get_Pos())

            # Adjust line capacity
            ob.set_Rate(ob.get_Rate()/self.data['baseMVA'])

        # Initialize security data for nodes
        for ob in self.Bus:
            ob.set_N1([None]*(self.data['SecurityNo']+1))
            ob.set_N1(ob.get_Pos(), 0)

        # Are all the loads the same type?
        aux = len(sett['Load_Type'])
        if aux == 1:
            if sett['Load_Type'][0] == 1:
                # An update is only needed if the loads are rural
                for ob in self.Bus:
                    ob.set_LT(sett['Load_Type'])
        elif aux > 1:
            # Update a set of the buses
            xb = 0
            for val in sett['Load_Type']:
                self.Bus[xb].set_LT(val)
                xb += 1

        # Enable branches in other scenarios (pyomo)
        xsec = 0
        for xs in self.data['N-1']:
            xsec += 1
            # Add N-1 information to buses
            for ob in self.Bus:
                ob.set_N1(xsec*self.data['Buses']+ob.data['Position'], xsec)

            # Add N-1 information to branches
            for ob in (self.Branch[xb] for xb in range(self.data['Branches'])
                       if xb+1 != xs):
                ob.set_N1(xcou, xsec)
                xcou += 1

        # Model losses
        if sett['Losses']:
            for ob in self.Bus:
                ob.set_LossF(ob.get_BraF())
                ob.set_LossT(ob.get_BraT())

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


class GenClass:
    ''' Core generation class '''
    def cNEBaseload_rule(self, m, xt, Tim, w, ConG):
        ''' Impose provision of baseload '''
        if self.data['Baseload'] > 0:
            return m.vNGen[ConG+self.pyomo['vNGen'], xt] >= \
                sum(m.vNGen[ConG+self.pyomo['vNGen'], xt2] for xt2 in Tim) * \
                self.data['Baseload']*w
        else:
            return Constraint.Skip

    def cNEGenC_rule(self, m, xc, xt, ConC, ConG, w):
        ''' Piece wise cost estimation '''
        (flg, x1,  x2, M1, M2) = self.cNEGenC_Auxrule(xc, ConC, ConG)
        if flg:
            return m.vNGCost[x1, xt]/w >= m.vNGen[x2, xt]*M1 + M2
        return Constraint.Skip

    def cNEGenC_Auxrule(self, xc, ConC, ConG):
        ''' Auxiliary for cNEGenC_rule '''
        flg = xc < self.pyomo['NoPieces']
        if flg:
            x1 = ConC+self.pyomo['vNGen']
            x2 = ConG+self.pyomo['vNGen']
            M1 = self.cost['LCost'][xc][0]
            M2 = self.cost['LCost'][xc][1]
        else:
            x1 = 0
            x2 = 0
            M1 = 0
            M2 = 0

        return flg, x1, x2, M1, M2

    def cNEGMax_rule(self, m, xt, ConG):
        ''' Maximum generation capacity '''
        return m.vNGen[ConG+self.pyomo['vNGen'], xt] <= self.data['Max']

    def cNEGMaxUC_rule(self, m, xt, ConG):
        ''' Maximum generation capacity '''
        if self.pyomo['vNGen_Bin'] is None:
            return m.vNGen[ConG+self.pyomo['vNGen'], xt] <= self.data['Max']

        return m.vNGen[ConG+self.pyomo['vNGen'], xt] <= self.data['Max'] * \
            m.vNGen_Bin[ConG+self.pyomo['vNGen_Bin'], xt]

    def cNEGMaxUCWC_rule(self, m, xt, ConG, Der):
        ''' Maximum generation capacity '''
        ''' Maximum generation capacity '''
        if self.pyomo['vNGen_Bin'] is None:
            return m.vNGen[ConG+self.pyomo['vNGen'], xt] <= self.data['Max']

        return m.vNGen[ConG+self.pyomo['vNGen'], xt] <= self.data['Max'] * \
            m.vNGen_Bin[ConG+self.pyomo['vNGen_Bin'], xt]*Der[xt]

    def cNEGMaxWC_rule(self, m, xt, ConG, Der):
        ''' Maximum generation capacity '''
        return m.vNGen[ConG+self.pyomo['vNGen'], xt] <= \
            self.data['Max']*Der[xt]

    def cNEGMin_rule(self, m, xt, ConG):
        ''' Minimum generation capacity - Only when needed '''
        if self.data['Max'] > 0:
            return m.vNGen[ConG+self.pyomo['vNGen'], xt] >= self.data['Min']
        else:
            return Constraint.Skip

    def cNEGMinDT1_rule(self, m, xt, xt0, ConG):
        ''' Minimum down time '''
        if self.pyomo['vNGen_MDT'] is None:
            return Constraint.Skip

        return m.vNGen_MDT[ConG+self.pyomo['vNGen_MDT'], xt] >= \
            m.vNGen_Bin[ConG+self.pyomo['vNGen_MDT'], xt0] - \
            m.vNGen_Bin[ConG+self.pyomo['vNGen_MDT'], xt]

    def cNEGMinDT2_rule(self, m, xt, LL, ConG):
        ''' Minimum down time '''
        if self.pyomo['vNGen_MDT'] is None:
            return Constraint.Skip

        return m.vNGen_MDT[ConG+self.pyomo['vNGen_MDT'], xt]*self.data['MDT'] \
            <= sum(1-m.vNGen_Bin[ConG+self.pyomo['vNGen_MDT'], xL] for xL in
                   LL[xt-1:xt+self.data['MDT']-1])

    def cNEGMinUC_rule(self, m, xt, ConG):
        ''' Minimum generation capacity - Only when needed '''
        if self.pyomo['vNGen_Bin'] is None:
            return Constraint.Skip

        return m.vNGen[ConG+self.pyomo['vNGen'], xt] >= \
            self.data['Min']*m.vNGen_Bin[ConG+self.pyomo['vNGen_Bin'], xt]

    def cNEGMinUCWC_rule(self, m, xt, ConG, Der):
        ''' Minimum generation capacity '''
        if self.pyomo['vNGen_Bin'] is None:
            return Constraint.Skip

        return m.vNGen[ConG+self.pyomo['vNGen'], xt] >= \
            self.data['Min']*m.vNGen_Bin[ConG+self.pyomo['vNGen_Bin'], xt] * \
            Der[xt]

    def cNEGMinUT1_rule(self, m, xt, xt0, ConG):
        ''' Minimum up time '''
        if self.pyomo['vNGen_MUT'] is None:
            return Constraint.Skip

        return m.vNGen_MUT[ConG+self.pyomo['vNGen_MUT'], xt] >= \
            m.vNGen_Bin[ConG+self.pyomo['vNGen_MUT'], xt] - \
            m.vNGen_Bin[ConG+self.pyomo['vNGen_MUT'], xt0]

    def cNEGMinUT2_rule(self, m, xt, LL, ConG):
        ''' Minimum up time - Only when needed '''
        if self.pyomo['vNGen_MUT'] is None:
            return Constraint.Skip

        return m.vNGen_MUT[ConG+self.pyomo['vNGen_MUT'], xt]*self.data['MUT'] \
            <= sum(m.vNGen_Bin[ConG+self.pyomo['vNGen_MUT'], xL] for xL in
                   LL[xt-1:xt+self.data['MUT']-1])

    def cNGenRampDown_rule(self, m, xt, xt0, ConG):
        ''' Generation ramps (down)'''
        if self.data['Ramp'] < self.data['Max']:
            aux = ConG+self.pyomo['vNGen']
            return m.vNGen[aux, xt]-m.vNGen[aux, xt0] <= self.data['Ramp']
        else:
            return Constraint.Skip

    def cNGenRampUp_rule(self, m, xt, xt0, ConG):
        ''' Generation ramps (down)'''
        if self.data['Ramp'] < self.data['Max']:
            aux = ConG+self.pyomo['vNGen']
            return m.vNGen[aux, xt0]-m.vNGen[aux, xt] <= self.data['Ramp']
        else:
            return Constraint.Skip

    def get_Bin(self):
        ''' get binaries for UC '''
        return self.pyomo['vNGen_Bin']

    def get_Bus(self):
        ''' Get bus number '''
        return self.data['Bus']

    def get_BusPos(self):
        ''' Get bus position '''
        return self.data['BusPosition']

    def get_Max(self):
        ''' Get maximum capacity (MW) '''
        return self.data['Max']

    def get_Min(self):
        ''' Get minimum capacity (MW) '''
        return self.data['Min']

    def get_NoPieces(self):
        ''' Get number of pices used for piece-wise cost estimations '''
        return self.pyomo['NoPieces']

    def get_P(self):
        ''' Get power output '''
        return self.data['PG']

    def get_Q(self):
        ''' Get reactive power output '''
        return self.data['QG']

    def get_Pos(self):
        ''' Get generator position '''
        return self.data['Position']

    def get_VG(self):
        ''' Get voltage magnitude'''
        return self.data['VG']

    def get_vNGen(self):
        ''' Set position of vNGen variable - pyomo'''
        return self.pyomo['vNGen']

    def initialise(self):
        ''' Finalize initialization '''
        # Adjust ramps
        if self.data['Ramp'] is None:
            self.data['Ramp'] = self.data['Max']+1
        elif self.data['Ramp'] <= 0 or self.data['Ramp'] >= 1:
            self.data['Ramp'] = self.data['Max']+1
        else:
            self.data['Ramp'] = self.data['Max']*self.data['Ramp']

    def set_Bin(self, xbin):
        ''' Set binaries for UC '''
        if self.data['Max'] > 0 and self.data['Min'] > 0:
            self.pyomo['vNGen_Bin'] = xbin
            xbin += 1
        else:
            self.pyomo['vNGen_Bin'] = None

        return xbin

    def set_MDT(self, xMDT):
        ''' Set variables for minimum down time '''
        if self.pyomo['vNGen_Bin'] is None or self.data['MDT'] is None:
            return xMDT

        self.pyomo['vNGen_MDT'] = xMDT
        xMDT += 1

        return xMDT

    def set_MUT(self, xMUT):
        ''' Set variables for minimum up time '''
        if self.pyomo['vNGen_Bin'] is None or self.data['MUT'] is None:
            return xMUT

        self.pyomo['vNGen_MUT'] = xMUT
        xMUT += 1

        return xMUT

    def set_CostCurve(self, sett, xNo, xLen, Base):
        ''' Define piece wise cost curve approximation '''

        if self.cost['MODEL'] == 1:
            # Piece-wise model
            NoPieces = self.cost['NCOST']
            xval = np.zeros(NoPieces, dtype=float)
            yval = np.zeros(NoPieces, dtype=float)
            xp = 0
            for x in range(NoPieces):
                xval[x] = self.cost['COST'][xp]
                yval[x] = self.cost['COST'][xp+1]
                xp += 2
            NoPieces -= 1
        else:
            # Polinomial model

            # Select number of pieces for the approximation
            if xLen == 0:  # Default case
                Delta = self.data['Max']
                Delta /= 3
            elif xLen == 1:  # Single value for all generators
                Delta = sett['Pieces'][0]
            else:  # Predefined number of pieces
                Delta = sett['Pieces'][xNo]

            NoPieces = int(np.floor(self.data['Max']/Delta))
            xval = np.zeros(NoPieces+1, dtype=float)
            yval = np.zeros(NoPieces+1, dtype=float)
            aux = self.data['Min']
            for xp in range(NoPieces+1):
                xval[xp] = aux
                xc = self.cost['NCOST']-1
                yval[xp] = self.cost['COST'][xc]
                for x in range(1, self.cost['NCOST']):
                    xc -= 1
                    yval[xp] += self.cost['COST'][xc]*xval[xp]**x
                aux += Delta

        # Convert to LP constraints
        self.cost['LCost'] = np.zeros((NoPieces, 2), dtype=float)
        for xv in range(NoPieces):
            self.cost['LCost'][xv][0] = (yval[xv+1]-yval[xv]) / (xval[xv+1] -
                                                                 xval[xv])
            self.cost['LCost'][xv][1] = \
                yval[xv]-xval[xv]*self.cost['LCost'][xv][0]
            self.cost['LCost'][xv][0] *= Base

        self.pyomo['NoPieces'] = NoPieces

    def set_Max(self, val):
        ''' Set maximum capacity (MW) '''
        self.data['Max'] = val

    def set_Min(self, val):
        ''' Set minimum capacity (MW) '''
        self.data['Min'] = val

    def set_PosB(self, xb):
        ''' Set position of the bus '''
        self.data['BusPosition'] = xb

    def set_vNGen(self, val):
        ''' Set position of vNGen variable - pyomo'''
        self.pyomo['vNGen'] = val


class Conventional(GenClass):
    ''' Conventional generator '''
    def __init__(self, obj):
        ''' Initialise generator class

        The class can use the following parameters:
        ['APF', 'MBASE', 'PC1', 'PC2', 'PG', 'QC1MIN', 'QC1MAX',
        'QC2MIN', 'QC2MAX', 'QG', 'QMAX', 'QMIN', 'RAMP_AGC',
        'RAMP_10', 'RAMP_30', 'RAMP_Q', 'RES', 'VG']
        However, only the ones that are currently used are passed
        ['COST', 'MODEL', 'NCOST', 'SHUTDOWN', 'STARTUP']
        '''
        # Parameters currently in use
        aux = ['Ancillary', 'Baseload', 'PMAX', 'PMIN', 'Ramp', 'Position',
               'VG', 'PG', 'QG', 'MDT', 'MUT']

        # Get settings
        self.data = {}
        self.data['Max'] = obj.settings['PMAX']
        self.data['Min'] = obj.settings['PMIN']
        for xa in aux:
            self.data[xa] = obj.settings[xa]
        self.data['Bus'] = obj.settings['GEN_BUS']

        aux = ['COST', 'MODEL', 'NCOST', 'SHUTDOWN', 'STARTUP']
        self.cost = {}
        for xa in aux:
            self.cost[xa] = obj.cost[xa]

        self.pyomo = {}
        self.pyomo['vNGen'] = None
        self.pyomo['vNGen_Bin'] = None
        self.pyomo['vNGen_MDT'] = None
        self.pyomo['vNGen_MUT'] = None
        self.pyomo['NoPieces'] = None


class Hydropower(GenClass):
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
        self.data['Min'] = 0
        for xa in aux:
            self.data[xa] = obj.settings[xa]

        aux = ['COST', 'MODEL', 'NCOST']
        self.cost = {}
        for xa in aux:
            self.cost[xa] = obj.cost[xa]

        self.pyomo = {}
        self.pyomo['vNGen'] = None
        self.pyomo['vNGen_Bin'] = None
        self.pyomo['vNGen_MDT'] = None
        self.pyomo['vNGen_MUT'] = None
        self.pyomo['NoPieces'] = None

    def get_Cost(self):
        ''' Return linear costs '''
        return self.cost['COST'][3]

    def cNEGMinDT1_rule(self, m, xt, xt0, ConG):
        ''' Minimum down time '''
        return Constraint.Skip

    def cNEGMinDT2_rule(self, m, xt, LL, ConG):
        ''' Minimum down time '''
        return Constraint.Skip

    def cNEGMinUT1_rule(self, m, xt, xt0, ConG):
        ''' Minimum up time '''
        return Constraint.Skip

    def cNEGMinUT2_rule(self, m, xt, LL, ConG):
        ''' Minimum up time '''
        return Constraint.Skip


class RES(GenClass):
    ''' RES generation '''
    def __init__(self, obj):
        ''' Initialise RES generator class

        The class can use the following parameters:
        ['Bus', 'Cost', 'Max', 'Uncertainty', 'Position']
        ['MODEL', 'NCOST', 'COST']
        However, only the ones that are currently used are passed
        '''
        # Parameters currently in use
        aux = ['Bus', 'Cost', 'Max', 'Uncertainty', 'Position']

        # Get settings
        self.data = {}
        self.data['Min'] = 0
        for xa in aux:
            self.data[xa] = obj.settings[xa]

        aux = ['MODEL', 'NCOST', 'COST']
        self.cost = {}
        for xa in aux:
            self.cost[xa] = obj.cost[xa]

        self.pyomo = {}
        self.pyomo['vNGen'] = None
        self.pyomo['NoPieces'] = None

    def cNEBaseload_rule(self, m, xt, Tim, w, ConG):
        ''' Impose provision of baseload '''
        return Constraint.Skip

    def cNEGMax_rule(self, m, xt, ConG):
        ''' Maximum generation capacity '''
        return Constraint.Skip

    def cNEGMaxUC_rule(self, m, xt, ConG):
        ''' Maximum generation capacity '''
        return Constraint.Skip

    def cNEGMin_rule(self, m, xt, ConG):
        ''' Minimum generation capacity '''
        return Constraint.Skip

    def cNEGMinDT1_rule(self, m, xt, xt0, ConG):
        ''' Minimum down time '''
        return Constraint.Skip

    def cNEGMinDT2_rule(self, m, xt, LL, ConG):
        ''' Minimum down time '''
        return Constraint.Skip

    def cNEGMinUC_rule(self, m, xt, ConG):
        ''' Minimum generation capacity '''
        return Constraint.Skip

    def cNEGMinUT1_rule(self, m, xt, xt0, ConG):
        ''' Minimum up time '''
        return Constraint.Skip

    def cNEGMinUT2_rule(self, m, xt, LL, ConG):
        ''' Minimum up time '''
        return Constraint.Skip

    def cNGenRampDown_rule(self, m, xt, xt0, ConG):
        ''' Generation ramps (down)'''
        return Constraint.Skip

    def cNGenRampUp_rule(self, m, xt, xt0, ConG):
        ''' Generation ramps (down)'''
        return Constraint.Skip

    def get_Bin(self):
        ''' get binaries for UC '''
        return None

    def get_Cost(self):
        ''' Return linear costs '''
        return self.cost['COST'][3]

    def set_Bin(self, xbin):
        ''' Set binaries for UC '''
        return xbin

    def set_MDT(self, xMDT):
        ''' Set variables for minimum down time '''
        return xMDT

    def set_MUT(self, xMUT):
        ''' Set variables for minimum up time '''
        return xMUT

    def initialise(self):
        ''' Finalize initialization '''
        return


class Generators:
    ''' Electricity generators '''
    def __init__(self, NoConv=0, NoHydro=0, NoRES=0):
        ''' General generator settings '''
        self.data = {
                'Conv': NoConv,
                'Hydro': NoHydro,
                'RES': NoRES,
                'Gen': NoConv+NoHydro+NoRES,
                'Types': None,
                'Bin': None
                }
        self.cooling = {}
        self.cooling['Flag'] = False
        self.cooling['Gen2Der'] = []
        self.cooling['Derate'] = None

        self.pyomo = {}
        self.pyomo['NoPieces'] = 0  # Max number of piece-wise cost curves
        self.pyomo['Type'] = []
        self.pyomo['Pos'] = []

        # Conventional generators
        self.ConvConf = [ConventionalConfig() for x in range(NoConv)]

        # Hydropower generators
        self.HydroConf = [HydropowerConfig() for x in range(NoHydro)]

        # RES generators
        self.RESConf = [RESConfig() for x in range(NoRES)]

    def cNEBaseload_rule(self, m, xg, xt, Tim, w, wF, ConG):
        ''' Baseload rule '''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].cNEBaseload_rule(m, xt, Tim, w[xt]/wF,
                                                      ConG)

    def _GClass(self, xg):
        ''' Get class and position of generator corresponsing to xg '''
        xa = self.data['Types'][self.pyomo['Type'][xg]]
        xp = self.pyomo['Pos'][xg]

        return (xa, xp)

    def cNEGenC_rule(self, m, xg, xc, xt, ConC, ConG, w):
        ''' Generation costs - Piece-wise estimation '''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].cNEGenC_rule(m, xc, xt, ConC, ConG, w)

    def cNEGenC_Auxrule(self, xg, xc, ConC, ConG):
        ''' Auxiliar for Generation costs - Piece-wise estimation '''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].cNEGenC_Auxrule(xc, ConC, ConG)

    def cNEGMax_rule(self, m, xg, xt, ConG):
        ''' Maximum generation capacity '''
        (xa, xp) = self._GClass(xg)
        # Considering water cooling constraints for conventional generators
        if self.cooling['Flag'] and self.pyomo['Type'][xg] == 0:
            aux = self.cooling['Derate'][self.cooling['Gen2Der'][xp]]
            return getattr(self, xa)[xp].cNEGMaxWC_rule(m, xt, ConG, aux)
        else:  # Ignoring water cooling constraints
            return getattr(self, xa)[xp].cNEGMax_rule(m, xt, ConG)

    def cNEGMaxUC_rule(self, m, xg, xt, ConG):
        ''' Maximum generation capacity '''
        (xa, xp) = self._GClass(xg)
        # Considering water cooling constraints for conventional generators
        if self.cooling['Flag'] and self.pyomo['Type'][xg] == 0:
            aux = self.cooling['Derate'][self.cooling['Gen2Der'][xp]]
            return getattr(self, xa)[xp].cNEGMaxUCWC_rule(m, xt, ConG, aux)
        else:  # Ignoring water cooling constraints
            return getattr(self, xa)[xp].cNEGMaxUC_rule(m, xt, ConG)

    def cNEGMin_rule(self, m, xg, xt, ConG):
        ''' Minimum generation capacity '''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].cNEGMin_rule(m, xt, ConG)

    def cNEGMinDT1_rule(self, m, xg, xt, xt0, ConG):
        ''' Minimum down time '''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].cNEGMinDT1_rule(m, xt, xt0, ConG)

    def cNEGMinDT2_rule(self, m, xg, xt, LL, ConG):
        ''' Minimum down time '''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].cNEGMinDT2_rule(m, xt, LL, ConG)

    def cNEGMinUC_rule(self, m, xg, xt, ConG):
        ''' Minimum generation capacity '''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].cNEGMinUC_rule(m, xt, ConG)

    def cNEGMinUT1_rule(self, m, xg, xt, xt0, ConG):
        ''' Minimum up time '''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].cNEGMinUT1_rule(m, xt, xt0, ConG)

    def cNEGMinUT2_rule(self, m, xg, xt, LL, ConG):
        ''' Minimum up time '''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].cNEGMinUT2_rule(m, xt, LL, ConG)

    def cNGenRampDown_rule(self, m, xg, xt, xt0, ConG):
        ''' Generation ramps (down)'''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].cNGenRampDown_rule(m, xt, xt0, ConG)

    def cNGenRampUp_rule(self, m, xg, xt, xt0, ConG):
        ''' Generation ramps (down)'''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].cNGenRampUp_rule(m, xt, xt0, ConG)

    def get_Bin(self, xg):
        ''' Get position of binaries for a generator '''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].get_Bin()

    def get_GenInBus(self, Bus):
        ''' Get list of generators connected to a bus - vNGen'''
        aux = []
        for xt, xp in zip(Bus.data['GenType'], Bus.data['GenPosition']):
            aux.append(getattr(self, self.data['Types'][xt])[xp].get_vNGen())

        return aux

    def get_GenAll(self):
        ''' Get all nodes - vNGen '''
        aux = self.get_GenDataAll('pyomo', 'vNGen')

        return aux

    def get_GenAllC(self):
        ''' Get RES nodes - vNGen '''
        aux = self.get_GenDataType('pyomo', 'vNGen', 'Conv')

        return aux

    def get_GenAllH(self):
        ''' Get RES nodes - vNGen '''
        aux = self.get_GenDataType('pyomo', 'vNGen', 'Hydro')

        return aux

    def get_GenAllR(self):
        ''' Get RES nodes - vNGen '''
        aux = self.get_GenDataType('pyomo', 'vNGen', 'RES')

        return aux

    def get_GenDataAll(self, at='data', dt='Bus'):
        ''' Get piece of data from all generators '''
        Dat = np.zeros(self.data['Gen'], dtype=int)
        x = 0
        for ax in self.data['Types']:
            for ob in getattr(self, ax):
                Dat[x] = getattr(ob, at)[dt]
                x += 1

        return Dat

    def get_GenDataType(self, at='data', dt='Bus', tp='Hydro'):
        ''' Get piece of data from specific generators '''
        Dat = np.zeros(self.data[tp], dtype=int)
        x = 0
        for ob in getattr(self, tp):
            Dat[x] = getattr(ob, at)[dt]
            x += 1

        return Dat

    def get_Max(self, xg):
        ''' Generation capacity'''
        (xa, xp) = self._GClass(xg)
        return getattr(self, xa)[xp].get_Max()

    def get_NoBin(self):
        ''' Number of binaries required for the generators '''
        return self.data['Bin']

    def get_NoCon(self):
        ''' Number of generators '''
        return self.data['Conv']

    def get_NoGen(self):
        ''' Number of generators '''
        return self.data['Gen']

    def get_NoHydro(self):
        ''' Number of hydropower units '''
        return self.data['Hydro']

    def get_NoMDT(self):
        ''' Number of binaries required for the generators '''
        return self.data['MDT']

    def get_NoMUT(self):
        ''' Number of binaries required for the generators '''
        return self.data['MUT']

    def get_NoPieces(self):
        ''' Return number of pieces for cost estimations  '''
        return self.pyomo['NoPieces']

    def get_NoRES(self):
        ''' Number of RES units '''
        return self.data['RES']

    def get_vNGenH(self, xg):
        ''' Get position of vNGen variable - pyomo/hydro'''
        return getattr(self, 'Hydro')[xg].get_vNGen()

    def get_vNGenR(self, xg):
        ''' Get position of vNGen variable - pyomo/hydro'''
        return getattr(self, 'RES')[xg].get_vNGen()

    def initialise(self, ENetwork, sett, RM):
        ''' Prepare objects and remove configuration versions '''

        # Get cooling information
        self.cooling['Flag'] = RM.cooling['Flag']
        if self.cooling['Flag']:
            # Link generators and capacity derate profiles
            self.cooling['Gen2Der'] = np.zeros(self.data['Conv'], dtype=int)
            for x in range(len(RM.cooling['Gen2Temp'])):
                self.cooling['Gen2Der'][x] = RM.cooling['Gen2Temp'][x]

            # NUmber of profiles and time periods
            NoT = len(RM.cooling['temp_w'][0])

            # Link generator types to cooling data
            aux = len(RM.cooling['GenType'])
            if aux == 0:
                # All generator types are the same
                NoW = len(RM.cooling['temp_w'])
                # Produce profiles
                self.cooling['Derate'] = np.zeros((NoW, NoT), dtype=float)
                for x in range(NoW):
                    self.cooling['Derate'][x] = \
                        RM.get_CF(temp_w=RM.cooling['temp_w'][x], index=0)
            else:
                # Check generator types
                axType = np.zeros(self.data['Conv'], dtype=int)
                for x in range(aux):
                    axType[x] = RM.cooling['GenType'][x]
                # Find number of simulations to be made,
                # i.e., unique combinations of temperatures and generator types
                aux = {}
                aux['No'] = 0
                aux['type'] = []
                aux['prof'] = []
                for x1 in range(self.data['Conv']):
                    x2 = 0
                    # Check if it is a new or repeated case
                    while x2 <= x1:
                        if x2 == x1:  # Is it the end of the loop?
                            # New profile
                            aux['type'].append(axType[x1])
                            aux['prof'].append(self.cooling['Gen2Der'][x1])
                            self.cooling['Gen2Der'][x1] = aux['No']
                            aux['No'] += 1
                        elif axType[x1] == axType[x2] and \
                                self.cooling['Gen2Der'][x1] == \
                                self.cooling['Gen2Der'][x2]:
                            # Same as x2
                            self.cooling['Gen2Der'][x1] = x2
                            x2 = x1
                        x2 += 1
                # Produce profiles
                self.cooling['Derate'] = \
                    np.zeros((aux['No'], NoT), dtype=float)
                for x in range(aux['No']):
                    self.cooling['Derate'][x] = \
                        RM.get_CF(temp_w=RM.cooling['temp_w'][aux['prof'][x]],
                                  index=aux['type'][x])

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

        # Initialise generators
        self.data['Types'] = ['Conv', 'Hydro', 'RES']
        xLen = len(sett['Pieces'])
        xbin = 0
        xMDT = 0
        xMUT = 0
        xt = 0
        xNo = 0
        for ax in self.data['Types']:
            xp = 0
            for ob in getattr(self, ax):
                # Link generators and buses
                xb = ENetwork.findBusPosition(ob.get_Bus())  # Bus
                xp = ob.data['Position']  # Generator
                # The Generator knows the position of its bus
                ob.set_PosB(xb)
                # The bus knows the type and location of the generator
                ENetwork.Bus[xb].add_Gen(xt, xp)

                # Add UC considerations
                xbin = ob.set_Bin(xbin)
                xMDT = ob.set_MDT(xMDT)
                xMUT = ob.set_MUT(xMUT)

                # Create cost curves
                ob.set_CostCurve(sett, xNo, xLen, ENetwork.get_Base())

                # MW --> pu
                ob.set_Max(ob.get_Max()/ENetwork.get_Base())
                ob.set_Min(ob.get_Min()/ENetwork.get_Base())

                # Store location of vGen variable
                ob.set_vNGen(xNo)

                # Link between positions an generation classes
                self.pyomo['Type'].append(xt)
                self.pyomo['Pos'].append(xp)

                # Store maximum number of pieces used so far
                aux = ob.get_NoPieces()
                if aux > self.pyomo['NoPieces']:
                    self.pyomo['NoPieces'] = aux

                # Finalise initialisation
                ob.initialise()

                xp += 1
                xNo += 1
            xt += 1
        self.data['Bin'] = xbin
        self.data['MUT'] = xMUT
        self.data['MDT'] = xMDT

    def MPCconfigure(self, mpc, conv, hydro, RES):
        ''' Initialize using mat power data '''

        # Conventional generators
        for x in range(self.data['Conv']):
            self.ConvConf[x].MPCconfigure(mpc, conv, x)

        for x in range(self.data['Hydro']):
            self.HydroConf[x].MPCconfigure(hydro, x)

        for x in range(self.data['RES']):
            self.RESConf[x].MPCconfigure(RES, x)

    def set_Max(self, xg, val):
        ''' Update generation capacity '''
        (xa, xp) = self._GClass(xg)
        getattr(self, xa)[xp].set_Max(val)
