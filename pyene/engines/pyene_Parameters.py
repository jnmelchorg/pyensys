"""
Created on Mon April 06 2020

A common template of parameters for power systems and temporal trees is provided
in this file. Most of the initial functions have been taken from the file pyeneD
created by Dr Eduardo Alejandro Martínez Ceseña.

@author: Dr. Jose Nicolas Melchor Gutierrez
         Dr Eduardo Alejandro Martínez Ceseña
"""

import math
import numpy as np
import copy
from pyomo.core import Constraint

'''                               DEVICE CLASSES                            '''

class Branch:
    ''' Electricity branch '''
    def __init__(self):
        # Basic data
        aux = ['ANGMAX', 'ANGMIN', 'BR_B', 'BR_R', 'BR_STATUS', 'BR_X',
               'Number', 'F_BUS', 'RATE_A', 'RATE_B', 'RATE_C', 'TAP',
               'T_BUS', 'Loss_Fix', 'Position', 'F_Position', 'T_Position',
               'N-1', ]
        self.data = {}
        for x in aux:
            self.data[x] = None

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
        return self.data['N-1'][x]

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
        return self.data['N-1'][xs]

    def get_Tap(self):
        ''' Get tap position '''
        return self.data['TAP']

    def get_X(self):
        ''' Get Reactance '''
        return self.data['BR_X']

    def is_active(self, xs):
        ''' Is the line connected in this scenario? '''
        return self.data['N-1'][xs] is not None

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
            self.data['N-1'] = val
        else:
            self.data['N-1'][x] = val

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
    def __init__(self):
        # Basic data
        aux = ['BASE_KV', 'BS', 'BUS_AREA', 'BUS_TYPE', 'BUS_X', 'BUS_Y',
               'Demand', 'GS', 'PeakP', 'PeakQ', 'Position', 'Name', 'Number',
               'VM', 'VA', 'VMAX', 'VMIN', 'ZONE', 'Load_Type', 'Loss_Fix',
               'N-1']
        self.data = {}
        for x in aux:
            self.data[x] = None

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
        return self.data['N-1'][xs]

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
            self.data['N-1'] = val
        else:
            self.data['N-1'][x] = val

    def set_LT(self, val):
        ''' Set load type (0:Urban, 1:Rural) '''
        self.data['Load_Type'] = val


class ElectricityNetwork:
    ''' Electricity network '''
    def __init__(self, NoBus=1, NoBranch=1, NoConv=0, NoHydro=0, NoRES=0):
        ''' General electricity network settings '''
        self.data = {
                'version': None,
                'baseMVA': None,
                'NoGen': None,
                'Slack': None,
                'Buses': NoBus,  # Number of buses
                'Branches': NoBranch,  # Number of branches
                'Security': None,  # N-1 cases to consider
                'SecurityNo': None,  # Number of N-1 cases
                'Conv' : NoConv, # Number of conventional generators
                'Hydro' : NoHydro, # Number of Hydro generators
                'RES' : NoRES # Number of RES generators
                }
        self.loss = {}
        self.loss['A'] = None
        self.loss['B'] = None

        # Initialise bus object
        self.Bus = [Bus() for x in
                    range(self.data['Buses'])]

        # Initialise branch object
        self.Branch = [Branch() for x in
                       range(self.data['Branches'])]
        
        # Initialise Conventional generator object
        self.Conv = [Conventional() for x in
                       range(self.data['Branches'])]

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
    
    def get_NoConv(self):
        ''' Get Number of Conv units '''
        return self.data['Conv']
    
    def get_NoHydro(self):
        ''' Get Number of Hydro units '''
        return self.data['Hydro']
    
    def get_NoRES(self):
        ''' Get Number of RES units '''
        return self.data['RES']

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
        aux = len(sett['Load_type'])
        if aux == 1:
            if sett['Load_type'][0] == 1:
                # An update is only needed if the loads are rural
                for ob in self.Bus:
                    ob.set_LT(sett['Load_type'])
        elif aux > 1:
            # Update a set of the buses
            xb = 0
            for val in sett['Load_type']:
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
    
    def set_NoConv(self, val):
        ''' Set Number of Conv units '''
        self.data['Conv'] = val
    
    def set_NoHydro(self, val):
        ''' Set Number of Hydro units '''
        self.data['Hydro'] = val
    
    def set_NoRES(self, val):
        ''' Set Number of RES units '''
        self.data['RES'] = val


class GenClass:
    ''' Core generation class '''
    def __init__(self):
        # Basic data
        aux = ['Ancillary', 'APF', 'GEN', 'Bus', 'MBASE', 'PC1', 'PC2',
               'PG', 'PMAX', 'PMIN', 'QC1MIN', 'QC1MAX', 'QC2MIN', 'QC2MAX',
               'QG', 'QMAX', 'QMIN', 'Ramp', 'RAMP_AGC', 'RAMP_10', 'RAMP_30',
               'RAMP_Q', 'RES', 'VG', 'MDT', 'MUT', 'Baseload', 'COST',
               'MODEL', 'NCOST', 'SHUTDOWN', 'STARTUP', 'Position', 'NoPieces',
               'LCost', 'BusPosition', 'UniCost', 'Uncertainty']
        self.data = {}
        for x in aux:
            self.data[x] = None

    def get_Bus(self):
        ''' Get bus number '''
        return self.data['Bus']

    def get_BusPos(self):
        ''' Get bus position '''
        return self.data['BusPosition']

    def get_PMax(self):
        ''' Get maximum capacity (MW) '''
        return self.data['PMAX']

    def get_PMin(self):
        ''' Get minimum capacity (MW) '''
        return self.data['PMIN']

    def get_NoPieces(self):
        ''' Get number of pieces used for piece-wise cost estimations '''
        return self.data['NoPieces']

    def get_P(self):
        ''' Get power output '''
        return self.data['PG']

    def get_Q(self):
        ''' Get reactive power output '''
        return self.data['QG']

    def get_Pos(self):
        ''' Get generator position '''
        return self.data['Position']
    
    def get_UniCost(self):
        ''' Return coefficient cost of linear generation cost'''
        return self.data['UniCost']

    def get_VG(self):
        ''' Get voltage magnitude'''
        return self.data['VG']

    def set_CostCurve(self, NoPieces, A, B):
        ''' Set parameters for piece wise cost curve approximation '''
        self.data['LCost'] = np.zeros((NoPieces, 2), dtype=float)
        for xv in range(NoPieces):
            self.data['LCost'][xv][0] = A[xv]
            self.data['LCost'][xv][1] = B[vx]

        self.data['NoPieces'] = NoPieces

    def set_PMax(self, val):
        ''' Set maximum capacity (MW) '''
        self.data['PMAX'] = val

    def set_PMin(self, val):
        ''' Set minimum capacity (MW) '''
        self.data['PMIN'] = val

    def set_PosB(self, xb):
        ''' Set position of the bus '''
        self.data['BusPosition'] = xb
    
    def set_Ramp(self, val):
        ''' Set Ramps'''
        self.data['Ramp'] = val
    
    def set_UniCost(self, val):
        ''' Set coefficient cost of linear generation cost'''
        self.data['UniCost'] = val


class Conventional(GenClass):
    ''' Conventional generator '''
    def __init__(self):
        self.cooling = {}
        self.cooling['Flag'] = False
        self.cooling['Gen2Der'] = []
        self.cooling['Derate'] = None
        super().__init__()


class Hydropower(GenClass):
    ''' Hydropower generator '''
    def __init__(self):
        super().__init__()


class RES(GenClass):
    ''' RES generation '''
    def __init__(self, obj):
        super().__init__()


class Generators:
    ''' Electricity generators '''

    def _GClass(self, xg):
        ''' Get class and position of generator corresponsing to xg '''
        xa = self.data['Types'][self.pyomo['Type'][xg]]
        xp = self.pyomo['Pos'][xg]

        return (xa, xp)

    def get_GenInBus(self, Bus):
        ''' Get list of generators connected to a bus - vNGen'''
        aux = []
        for xt, xp in zip(Bus.data['GenType'], Bus.data['GenPosition']):
            aux.append(getattr(self, self.data['Types'][xt])[xp].get_vNGen())

        return aux

    def initialise(self, ENetwork, sett, RM):
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