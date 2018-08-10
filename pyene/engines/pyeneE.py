# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:04:58 2018

Pyene Energy provides methods for balancing multiple energy vectors at
different time aggregation levels

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
# convert in or long into float before performing divissions
from __future__ import division

# Make pyomo symbols known to python
from pyomo.core import Constraint, Var, NonNegativeReals
import numpy as np
import json
import os


class pyeneEConfig:
    ''' Default settings used for this class '''
    def __init__(self):
        # Default time-step and map
        self.data = {}
        # Monthly resolution
        NoVec = 3
        aux = np.zeros(NoVec, dtype=float)
        for x in range(NoVec):
            aux[x] = 5+x
        self.data["0"] = {
                "Title": "Month",
                "Names": ["typical"],
                "Weights": [1],
                "Inputs": aux,
                "Outputs": np.zeros(NoVec, dtype=float),
                "Uncertainty": False
                }
        # Typical number of weeks in a month
        self.data["1"] = {
                "Title": "Weeks",
                "Names": ["typical"],
                "Weights": [4.3333],
                "Inputs": np.zeros(NoVec, dtype=float),
                "Outputs": np.zeros(NoVec, dtype=float),
                "Uncertainty": False
                }
        # Representation of a week
        self.data["2"] = {
                "Title": "Days",
                "Names": ["Weekday", "Weekend"],
                "Weights": [5, 2],
                "Inputs": np.zeros((NoVec, 2), dtype=float),
                "Outputs": np.zeros((NoVec, 2), dtype=float),
                "Uncertainty": False
                }
        # CONTROL SETTINGS AND OTHER INTERNAL DATA
        self.settings = {
                'File': None,
                'Fix': False,  # Force a specific number of vectors
                'Vectors': NoVec  # Number of vectors
                }
        # Size of data
        self.size = {
                'Periods': [],  # Number of periods
                'LenPeriods': [],  # Length of each period
                'SumPeriods': [],  # Sum of period lengths
                'Scenarios': [],  # Number of scenarios
                'Vectors': [],  # Number of vectors
                'Nodes': []  # Number of nodes
                }


class EnergyClass:
    def __init__(self, obj=None):
        ''' Initialise network class '''
        # Get default values
        if obj is None:
            obj = pyeneEConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

        # sets and parameters used for the mathematical model
        self.s = {}
        self.p = {}

    def _Connect(self, Unc):
        ''' Produce connectivity matrices '''
        # Build main LL defining scenario tree connections
        LL_TimeNodeTree = np.zeros((self.size['Nodes'], 2), dtype=int)
        LL_TimeScenarioTree = np.zeros((self.size['Nodes'], 2), dtype=int)
        LL_TimeNodeTree[0][0] = 1
        LL_TimeNodeTree[0][1] = self.size['LenPeriods'][0]
        LL_TimeScenarioTree[0][1] = self.size['Scenarios']-1

        # Beginning of each time level
        LL_TimeLevel = np.ones(self.size['Periods']+1, dtype=int)

        # Number of branches spaning from a node
        aux1 = 1
        # Total number of nodes
        aux2 = self.size['LenPeriods'][0]
        # Position in the linked list
        x1 = 0
        for x2 in range(self.size['Periods']-1):
            # Get position of each level of the tree
            LL_TimeLevel[x2+1] = LL_TimeLevel[x2] + aux1
            # Calculate number of new branches
            aux1 = aux1*self.size['LenPeriods'][x2]
            # Link to scenarios (beginning from zero)
            aux3 = 0
            for x3 in range(aux1):
                x1 = x1+1
                LL_TimeNodeTree[x1][0] = aux2 + 1
                LL_TimeNodeTree[x1][1] = aux2 + self.size['LenPeriods'][x2+1]
                aux2 += self.size['LenPeriods'][x2+1]
                LL_TimeScenarioTree[x1][0] = aux3
                LL_TimeScenarioTree[x1][1] = (aux3 + self.size['Scenarios'] /
                                              aux1-1)
                aux3 += self.size['Scenarios']/aux1

        # Beginning of the final time level
        LL_TimeLevel[self.size['Periods']] = (LL_TimeLevel
                                              [self.size['Periods']-1]+aux1)

        # Get nodes at each time level
        NodeTime = np.zeros((self.size['Periods']+1, 2), dtype=int)
        xacu1 = 0
        xacu2 = 1
        for xt in range(self.size['Periods']):
            xacu2 *= self.size['LenPeriods'][xt]
            NodeTime[xt+1][:] = [xacu1+1, xacu1+xacu2]
            xacu1 += xacu2

        # Adding cnnection to the last set of scenarios
        aux1 = aux1*self.size['LenPeriods'][self.size['Periods']-1]
        aux3 = 0
        for x3 in range(aux1):
            x1 = x1+1
            LL_TimeScenarioTree[x1][0] = aux3
            LL_TimeScenarioTree[x1][1] = aux3 + self.size['Scenarios']/aux1-1
            aux3 = aux3 + self.size['Scenarios']/aux1

        # Identify node connections in the previous period
        LL_TimeNodeBefore = np.zeros(self.size['Nodes'], dtype=int)
        for x1 in range(LL_TimeLevel[self.size['Periods']]-1):
            # print(x1)
            x2 = LL_TimeNodeTree[x1, 0]-1
            while x2 <= LL_TimeNodeTree[x1, 1]-1:
                x2 += 1
                LL_TimeNodeBefore[x2] = x1

        # Identify node connection based on sequence of the tree
        saux = LL_TimeLevel[self.size['Periods']]-1
        xin = 1
        xlvl = 1
        xbefore = [0, 0]
        # [1:2] Node before when moving forward
        # [3;4] Node before when moving backwards
        LL_TimeNodeBeforeSequence = np.zeros((self.size['Nodes'], 4),
                                             dtype=int)
        LL_TimeNodeBeforeSequence, x = self.Mapping(xin, xlvl, xbefore,
                                                    LL_TimeNodeBeforeSequence,
                                                    LL_TimeNodeTree, Unc, saux)

        # Mark nodes that face uncertainty
        LL_Unc = np.zeros(self.size['Nodes'], dtype=int)
        for x1 in range(self.size['Periods']):
            if Unc[x1] == 1:
                x2 = LL_TimeLevel[x1]-1
                while x2 <= LL_TimeLevel[x1+1]-2:
                    LL_Unc[x2] = 1
                    x2 += 1

        # Add Weights
        LL_WIO = np.zeros(self.size['Nodes'], dtype=int)
        xacu1 = 1
        xacu2 = 0
        xacu3 = 1
        for x1 in range(self.size['Periods']):
            for x2 in range(xacu1):
                for x3 in range(self.size['LenPeriods'][x1]):
                    xacu2 += 1
                    LL_WIO[xacu2] = xacu3+x3
            xacu1 *= self.size['LenPeriods'][x1]
            xacu3 += self.size['LenPeriods'][x1]

        # Store outputs
        self.tree = {
                'Before': LL_TimeNodeBeforeSequence,  # Node before
                'After': LL_TimeNodeTree,  # Node after
                'Scenarios': LL_TimeScenarioTree,  # Scenarios
                'Uncertainty': LL_Unc,  # Flag for uncertainty
                'InOut': LL_WIO,  # Location of inputs/outputs
                'Time': NodeTime,  # Nodes in each time level
                }

    def _Link(self, Unc):
        ''' Build linked lists '''
        # Build first linked lists, forward connections for energy balance
        LL1 = np.zeros((self.size['Nodes'], 2), dtype=int)
        for x1 in range(1, self.size['Nodes']):
            for x2 in range(2):
                LL1[x1][x2] = self.tree['Before'][x1][x2]

        # Build linked lists for aggregation and uncertainty
        NosNod = self.size['Nodes']-1
        if sum(Unc) == 0:
            NosLL2 = NosNod
            NosLL3 = 0
            LL3 = np.zeros((NosLL3, 3), dtype=int)
        else:
            NosLL3 = sum(self.tree['Uncertainty'])-1
            NosLL2 = self.size['Nodes']-NosLL3-2
            LL3 = np.zeros((NosLL3+1, 3), dtype=int)
        LL2 = np.zeros((NosLL2+1, 3), dtype=int)

        x2 = 0
        x3 = 0
        for x1 in range(1, self.size['Nodes']):
            if self.tree['Uncertainty'][x1] == 0:
                x2 += 1
                LL2[x2][0] = x1
                LL2[x2][1] = self.tree['Before'][x1][2]
                LL2[x2][2] = self.tree['Before'][x1][3]
            else:
                LL3[x3][0] = x1
                LL3[x3][1] = self.tree['After'][x1][0]
                LL3[x3][2] = (self.tree['After'][x1][1] -
                              self.tree['After'][x1][0])
                x3 += 1

        # Linked lists
        self.LL = {
                'NosBal': NosNod,  # Number of nodes for energy balance
                'NosAgg': NosLL2,  # Number of rows for aggregation LL
                'NosUnc': NosLL3  # Number of rows for Uncertainty LL
                }
        self.p['LLTS1'] = LL1  # Linke list for energy balance
        self.p['LLTS2'] = LL2  # Aggregation
        self.p['LLTS3'] = LL3  # Weighted sum

    def _Measure(self):
        ''' Measure the size of the data arrays '''
        # Get number of time periods
        s_LL_time = len(self.data)

        sv_LL_time = np.zeros(s_LL_time, dtype=int)
        s_LL_timeVar = 1
        s_LL_timeSum = 1
        for x1 in range(s_LL_time):
            sv_LL_time[x1] = len(self.data[str(x1)]["Names"])
            s_LL_timeVar *= sv_LL_time[x1]
            s_LL_timeSum += sv_LL_time[x1]

        # Measuring the data
        aux = np.asarray(self.data[chr(48)]["Inputs"])
        if aux.ndim == 1:
            NosVec = len(self.data[chr(48)]["Inputs"])
        else:
            NosVec = len(aux)

        s_LL_Nodes = 1
        aux = 1
        for x1 in range(s_LL_time):
            aux = aux*sv_LL_time[x1]
            s_LL_Nodes = s_LL_Nodes + aux

        # Storing data
        self.size = {
                'Periods': s_LL_time,  # Number of periods
                'LenPeriods': sv_LL_time,  # Length of each period
                'SumPeriods': s_LL_timeSum,  # Sum of period lengths
                'Scenarios': s_LL_timeVar,  # Number of scenarios
                'Vectors': NosVec,  # Number of vectors
                'Nodes': s_LL_Nodes  # Number of nodes
                }
        # Data tree
        self.tree = {
                'Before': [],  # Node before
                'After': [],  # Node after
                'Scenarios': [],  # Scenarios
                'Uncertainty': [],  # Flag for uncertainty
                'InOut': [],  # Location of inputs/outputs
                'Time': [],  # Nodes in each time level
                }
        # Weights
        self.Weight = {
                'In': [],  # Weight of an input
                'Out': [],  # Weight of an output
                }

    def _Parameters(self, WIn, WOut, Wght):
        ''' Build parameters for optimisation.
        This information should ultimately be included in the input file
        '''
        WInFull = np.zeros((self.size['Nodes'], self.size['Vectors']),
                           dtype=float)
        WOutFull = np.zeros((self.size['Nodes'], self.size['Vectors']),
                            dtype=float)
        WghtFull = np.ones(self.size['Nodes'], dtype=float)

        for x1 in range(self.size['Nodes']):
            WghtFull[x1] = Wght[self.tree['InOut'][x1]]
            for xv in range(self.size['Vectors']):
                WInFull[x1][xv] = WIn[self.tree['InOut'][x1]][xv]
                WOutFull[x1][xv] = WOut[self.tree['InOut'][x1]][xv]
        self.Weight = {
                'In': WInFull,  # All inputs
                'Out': WOutFull  # All outputs
                }
        self.p['WghtFull'] = WghtFull  # Weight of each node

    def _Process(self):
        ''' Process the data as inputs, outputs, weights and uncertainty '''
        WIn = np.zeros((self.size['SumPeriods'], self.size['Vectors']),
                       dtype=float)
        WOut = np.zeros((self.size['SumPeriods'], self.size['Vectors']),
                        dtype=float)
        Wght = np.ones(self.size['SumPeriods'], dtype=float)
        Unc = np.zeros(self.size['Periods'], dtype=bool)

        acu = 1
        for x1 in range(self.size['Periods']):
            xtxt = str(x1)
            auxI = np.asarray(self.data[xtxt]["Inputs"])
            auxO = np.asarray(self.data[xtxt]["Outputs"])
            auxW = self.data[xtxt]["Weights"]
            auxs = len(auxW)
            auxU = self.data[xtxt]["Uncertainty"]
            Unc[x1] = auxU
            for x2 in range(auxs):
                Wght[acu+x2] = auxW[x2]
                if self.size['Vectors'] == 1:
                    if auxI.ndim == 1:
                        WIn[acu+x2][0] = auxI[x2]
                        WOut[acu+x2][0] = auxO[x2]
                    else:
                        WIn[acu+x2][0] = auxI[0][x2]
                        WOut[acu+x2][0] = auxO[0][x2]
                else:
                    if auxI.ndim == 1:
                        for xv in range(self.size['Vectors']):
                            WIn[acu+x2][xv] = auxI[xv]
                            WOut[acu+x2][xv] = auxO[xv]
                    else:
                        for xv in range(self.size['Vectors']):
                            WIn[acu+x2][xv] = auxI[xv][x2]
                            WOut[acu+x2][xv] = auxO[xv][x2]
            acu += auxs

        return WIn, WOut, Wght, Unc

    def addCon(self, m):
        ''' Adding pyomo constraints '''
        # Initialisation conditions
        m.cZSoC = Constraint(range(2), self.s['Vec'], rule=self.cZSoC_rule)
        # Balance at different time levels
        m.cSoCBalance = Constraint(self.s['Nodz'], self.s['Vec'],
                                   rule=self.cSoCBalance_rule)
        # Aggregating (deterministic case)
        m.cSOCAggregate = Constraint(self.s['LLTS2'], self.s['Vec'],
                                     rule=self.cSoCAggregate_rule)
        # Aggregating (stochastic case)
        if self.s['FUnc'] != 0:
            m.cSoCStochastic = Constraint(self.s['LLTS3'], self.s['Vec'],
                                          rule=self.cSoCStochastic_rule)

        return m

    def addPar(self, m):
        ''' Adding pyomo parameters '''

        # Nodal inputs and outputs (may be redefined as variables)
        m.WInFull = self.Weight['In']
        m.WOutFull = self.Weight['Out']

        return m

    def addSets(self, m):
        ''' Adding pyomo sets '''
        self.s['Nod'] = range(1, self.LL['NosBal']+1)
        self.s['Nodz'] = range(self.LL['NosBal']+1)
        self.s['LLTS2'] = range(self.LL['NosAgg']+1)
        self.s['LLTS3'] = range(self.LL['NosUnc']+1)
        self.s['FUnc'] = self.LL['NosUnc']
        self.s['Vec'] = range(self.size['Vectors'])

        return m

    def addVars(self, m):
        ''' Adding pyomo varaibles '''
        m.vSoC = Var(self.s['Nodz'], range(2), self.s['Vec'],
                     domain=NonNegativeReals, initialize=0.0)

        return m

    def cSoCAggregate_rule(self, m, xL2, xv):
        ''' Aggregating (deterministic case) '''
        return (m.vSoC[self.p['LLTS2'][xL2, 0], 1, xv] ==
                m.vSoC[self.p['LLTS2'][xL2, 1], self.p['LLTS2'][xL2, 2], xv] *
                self.p['WghtFull'][self.p['LLTS2'][xL2, 0]] +
                m.vSoC[self.p['LLTS1'][self.p['LLTS2'][xL2, 0], 0],
                       self.p['LLTS1'][self.p['LLTS2'][xL2, 0], 1], xv] *
                (1-self.p['WghtFull'][self.p['LLTS2'][xL2, 0]]))

    def cSoCBalance_rule(self, m, xL1, xv):
        ''' Balance at different time levels '''
        return (m.vSoC[xL1, 0, xv] ==
                m.vSoC[self.p['LLTS1'][xL1, 0], self.p['LLTS1'][xL1, 1], xv] +
                m.WInFull[xL1, xv] - m.WOutFull[xL1, xv])

    def cSoCStochastic_rule(self, m, xL3, xv):
        ''' Aggregating (stochastic case) '''
        return (m.vSoC[self.p['LLTS3'][xL3, 0], 1, xv] ==
                m.vSoC[self.p['LLTS1'][self.p['LLTS3'][xL3, 0], 0],
                       self.p['LLTS1'][self.p['LLTS3'][xL3, 0], 1], xv] *
                (1-self.p['WghtFull'][self.p['LLTS3'][xL3, 0]]) +
                self.p['WghtFull'][self.p['LLTS3'][xL3, 0]] *
                (m.vSoC[self.p['LLTS3'][xL3, 0], 0, xv] *
                 -self.p['LLTS3'][xL3, 2] +
                 sum(m.vSoC[self.p['LLTS3'][xL3, 1]+x1, 1, xv]
                     for x1 in range(self.p['LLTS3'][xL3, 2]+1))))

    def cZSoC_rule(self, m, x1, xv):
        ''' SoC initialisation conditiona '''
        return m.vSoC[0, x1, xv] == 0

    def initialise(self):
        ''' Initialise externally '''
        # Should a file be loaded?
        if self.settings['File'] is not None:
            print(self.settings['File'])
            self.data = json.load(open(self.settings['File']))

        # Measure the size of the data arrays
        self._Measure()

        # Adjust number of vectors and clear inputs and outputs
        if self.settings['Fix']:
            for xp in range(self.size['Periods']):
                if self.size['LenPeriods'][xp] == 1:
                    aux = np.zeros(self.settings['Vectors'], dtype=float)
                elif self.settings['Vectors'] == 1:
                    aux = np.zeros(self.size['LenPeriods'][xp], dtype=float)
                else:  # to a matrix
                    aux = np.zeros((self.settings['Vectors'],
                                    self.size['LenPeriods'][xp]), dtype=float)
                self.data[str(xp)]['Inputs'] = aux
                self.data[str(xp)]['Outputs'] = aux
            self.size['Vectors'] = self.settings['Vectors']

        # Summarize the data as inputs, outputs, weights and uncertainty
        (WIn, WOut, Wght, Unc) = self._Process()

        # Produce connectivity matrices and weights
        self._Connect(Unc)

        # Define inputs, outpurs and weights per node
        self._Parameters(WIn, WOut, Wght)
        self._Link(Unc)

    # Map the scenario tree (recursive function)
    def Mapping(self, xin, xlvl, xbeforeOld, LL_TimeNodeBeforeSequence,
                dta, Unc, s):
        xbeforeNew = xbeforeOld
        x1 = dta[xin-1, 0] - 1
        while x1 < dta[xin-1, 1]:
            x1 += 1
            # Hold node for future nodes
            LL_TimeNodeBeforeSequence[x1, 0] = xbeforeOld[0]
            LL_TimeNodeBeforeSequence[x1, 1] = xbeforeOld[1]
            xbeforeNew = [x1, 0]

            if x1+1 <= s:
                (LL_TimeNodeBeforeSequence,
                 xbeforeNew) = self.Mapping(x1+1, xlvl+1, xbeforeNew,
                                            LL_TimeNodeBeforeSequence, dta,
                                            Unc, s)

            LL_TimeNodeBeforeSequence[x1, 2] = xbeforeNew[0]
            LL_TimeNodeBeforeSequence[x1, 3] = xbeforeNew[1]
            xbeforeNew = [x1, 1]

            # Hold node for deterministic studies
            if Unc[xlvl-1] == 0:
                xbeforeOld = xbeforeNew

        return LL_TimeNodeBeforeSequence, xbeforeNew

    def print(self, m):
        ''' Print results '''
        for xv in self.s['Vec']:
            print('Vector No:', xv)
            for x1 in self.s['Nodz']:
                print("SoC[%3.0f" % x1, "][0:1]=[%10.2f"
                      % m.vSoC[x1, 0, xv].value, ", %10.2f"
                      % m.vSoC[x1, 1, xv].value, "]")

    def Read(self, FileName, jsonPath):
        ''' Read input data '''
        MODEL_JSON = os.path.join(jsonPath, FileName)
        data = json.load(open(MODEL_JSON))

        return data

    def OF_rule(self, m):
        ''' Objective function '''
        return sum(m.vSoC[1, 0, 0]-m.vSoC[3, 0, xv] for xv in self.s['Vec'])
