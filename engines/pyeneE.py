# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:04:58 2018

@author: mchihem2
"""
# convert in or long into float before performing divissions
from __future__ import division

# Make pyomo symbols known to python
from pyomo.environ import *
from pyomo.core import *
from pyomo.opt import SolverFactory

import numpy as np
import networkx as nx
import json
import os


# Linked lists
class EnergyClass:
    # Initialize
    def __init__(self):
        # Read from file
        self.fRea = False
        # Default time-step and map
        self.Input_Data = {}
        # Monthly resolution
        NoVec = 3
        aux = np.zeros(NoVec, dtype=float)
        for x in range(NoVec):
            aux[x] = 5+x
        self.Input_Data["0"] = {
                "Title": "Month",
                "Names": ["typical"],
                "Weights": [1],
                "Inputs": aux,
                "Outputs": np.zeros(NoVec, dtype=float),
                "Uncertainty": False
                }
        # Typical number of weeks in a month
        self.Input_Data["1"] = {
                "Title": "Weeks",
                "Names": ["typical"],
                "Weights": [4.3333],
                "Inputs": np.zeros(NoVec, dtype=float),
                "Outputs": np.zeros(NoVec, dtype=float),
                "Uncertainty": False
                }
        # Representation of a week
        self.Input_Data["2"] = {
                "Title": "Days",
                "Names": ["Weekday", "Weekend"],
                "Weights": [5, 2],
                "Inputs": np.zeros((NoVec, 2), dtype=float),
                "Outputs": np.zeros((NoVec, 2), dtype=float),
                "Uncertainty": False
                }        

    # Read input data
    def Read(self, FileName):
        MODEL_JSON = os.path.join(os.path.dirname(__file__), '..\json',
                                  FileName)
        Input_Data = json.load(open(MODEL_JSON))

        return Input_Data

    # Measure the size of the data arrays
    def _Measure(self):
        # Get number of time periods
        s_LL_time = len(self.Input_Data)

        sv_LL_time = np.zeros(s_LL_time, dtype=int)
        s_LL_timeVar = 1
        s_LL_timeSum = 1
        for x1 in range(s_LL_time):
            sv_LL_time[x1] = len(self.Input_Data[chr(48+x1)]["Names"])
            s_LL_timeVar *= sv_LL_time[x1]
            s_LL_timeSum += sv_LL_time[x1]

        # Measuring the data
        aux = np.asarray(self.Input_Data[chr(48)]["Inputs"])
        if aux.ndim == 1:
            NosVec = len(self.Input_Data[chr(48)]["Inputs"])
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

    # Process the data as inputs, outputs, weights and uncertainty
    def Process(self, s_LL_timeVar, s_LL_timeSum):
        WIn = np.zeros((s_LL_timeSum, self.size['Vectors']), dtype=int)
        WOut = np.zeros((s_LL_timeSum, self.size['Vectors']), dtype=float)
        Wght = np.ones(s_LL_timeSum, dtype=float)
        Unc = np.zeros(self.size['Periods'], dtype=bool)

        acu = 1
        for x1 in range(self.size['Periods']):
            xtxt = chr(48+x1)
            auxI = np.asarray(self.Input_Data[xtxt]["Inputs"])
            auxO = np.asarray(self.Input_Data[xtxt]["Outputs"])
            auxW = self.Input_Data[xtxt]["Weights"]
            auxs = len(auxW)
            auxU = self.Input_Data[xtxt]["Uncertainty"]
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

    # Produce connectivity matrices
    def Connect(self, sv_LL_time, s_LL_timeVar, Unc, s_LL_Nodes):
        # Build main LL defining scenario tree connections
        LL_TimeNodeTree = np.zeros((s_LL_Nodes, 2), dtype=int)
        LL_TimeScenarioTree = np.zeros((s_LL_Nodes, 2), dtype=int)
        LL_TimeNodeTree[0][0] = 1
        LL_TimeNodeTree[0][1] = sv_LL_time[0]
        LL_TimeScenarioTree[0][1] = s_LL_timeVar-1

        # Beginning of each time level
        LL_TimeLevel = np.ones(self.size['Periods']+1, dtype=int)

        # Number of branches spaning from a node
        aux1 = 1
        # Total number of nodes
        aux2 = sv_LL_time[0]
        # Position in the linked list
        x1 = 0
        for x2 in range(self.size['Periods']-1):
            # Get position of each level of the tree
            LL_TimeLevel[x2+1] = LL_TimeLevel[x2] + aux1
            # Calculate number of new branches
            aux1 = aux1*sv_LL_time[x2]
            # Link to scenarios (beginning from zero)
            aux3 = 0
            for x3 in range(aux1):
                x1 = x1+1
                LL_TimeNodeTree[x1][0] = aux2 + 1
                LL_TimeNodeTree[x1][1] = aux2 + sv_LL_time[x2+1]
                aux2 += sv_LL_time[x2+1]
                LL_TimeScenarioTree[x1][0] = aux3
                LL_TimeScenarioTree[x1][1] = aux3 + s_LL_timeVar/aux1-1
                aux3 += s_LL_timeVar/aux1

        # Beginning of the final time level
        LL_TimeLevel[self.size['Periods']] = LL_TimeLevel[self.size['Periods']-1]+aux1

        # Get nodes at each time level
        NodeTime = np.zeros((self.size['Periods']+1, 2), dtype=int)
        xacu1 = 0
        xacu2 = 1
        for xt in range(self.size['Periods']):
            xacu2 *= sv_LL_time[xt]
            NodeTime[xt+1][:] = [xacu1+1, xacu1+xacu2]
            xacu1 += xacu2

        # Adding cnnection to the last set of scenarios
        aux1 = aux1*sv_LL_time[self.size['Periods']-2]
        aux3 = 0
        for x3 in range(aux1):
            x1 = x1+1
            LL_TimeScenarioTree[x1][0] = aux3
            LL_TimeScenarioTree[x1][1] = aux3 + s_LL_timeVar/aux1-1
            aux3 = aux3 + s_LL_timeVar/aux1

        # Identify node connections in the previous period
        LL_TimeNodeBefore = np.zeros(s_LL_Nodes, dtype=int)
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
        LL_TimeNodeBeforeSequence = np.zeros((s_LL_Nodes, 4), dtype=int)
        LL_TimeNodeBeforeSequence, x = self.Mapping(xin, xlvl, xbefore,
                                                    LL_TimeNodeBeforeSequence,
                                                    LL_TimeNodeTree, Unc, saux)

        # Mark nodes that face uncertainty
        LL_Unc = np.zeros(s_LL_Nodes, dtype=int)
        for x1 in range(self.size['Periods']):
            if Unc[x1] == 1:
                x2 = LL_TimeLevel[x1]-1
                while x2 <= LL_TimeLevel[x1+1]-2:
                    LL_Unc[x2] = 1
                    x2 += 1

        # Add Weights
        LL_WIO = np.zeros(s_LL_Nodes, dtype=int)
        xacu1 = 1
        xacu2 = 0
        xacu3 = 1
        for x1 in range(self.size['Periods']):
            for x2 in range(xacu1):
                for x3 in range(sv_LL_time[x1]):
                    xacu2 += 1
                    LL_WIO[xacu2] = xacu3+x3
            xacu1 *= sv_LL_time[x1]
            xacu3 += sv_LL_time[x1]

        # Outputs
        return (LL_TimeNodeTree, LL_TimeScenarioTree, LL_Unc, LL_WIO,
                LL_TimeNodeBeforeSequence, LL_TimeLevel, NodeTime)

    # Build parameters for optimisation
    # This information should ultimately be included in the input file
    def Parameters(self, s_LL_Nodes, WIn, WOut, Wght, LL_WIO):
        WInFull = np.zeros((s_LL_Nodes, self.size['Vectors']), dtype=float)
        WOutFull = np.zeros((s_LL_Nodes, self.size['Vectors']), dtype=float)
        WghtFull = np.ones(s_LL_Nodes, dtype=float)

        for x1 in range(s_LL_Nodes):
            WghtFull[x1] = Wght[LL_WIO[x1]]
            for xv in range(self.size['Vectors']):
                WInFull[x1][xv] = WIn[LL_WIO[x1]][xv]
                WOutFull[x1][xv] = WOut[LL_WIO[x1]][xv]

        return WInFull, WOutFull, WghtFull

    # Build linked lists
    def Link(self, Unc, LL_Unc, s_LL_Nodes, LL_TimeNodeBeforeSequence):
        # Build first linked lists, forward connections for energy balance
        LL1 = np.zeros((s_LL_Nodes, 2), dtype=int)
        for x1 in range(1, s_LL_Nodes):
            for x2 in range(2):
                LL1[x1][x2] = LL_TimeNodeBeforeSequence[x1][x2]

        # Build linked lists for aggregation and uncertainty
        NosNod = s_LL_Nodes-1
        if sum(Unc) == 0:
            NosLL2 = NosNod
            NosLL3 = 0
            LL3 = np.zeros((NosLL3, 3), dtype=int)
        else:
            NosLL3 = sum(LL_Unc)-1
            NosLL2 = s_LL_Nodes-NosLL3-2
            LL3 = np.zeros((NosLL3+1, 3), dtype=int)
        LL2 = np.zeros((NosLL2+1, 3), dtype=int)

        x2 = 0
        x3 = 0
        for x1 in range(1, s_LL_Nodes):
            if LL_Unc[x1] == 0:
                x2 += 1
                LL2[x2][0] = x1
                LL2[x2][1] = LL_TimeNodeBeforeSequence[x1][2]
                LL2[x2][2] = LL_TimeNodeBeforeSequence[x1][3]
            else:
                LL3[x3][0] = x1
                LL3[x3][1] = self.LL_TimeNodeTree[x1][0]
                LL3[x3][2] = self.LL_TimeNodeTree[x1][1]-self.LL_TimeNodeTree[x1][0]
                x3 += 1

        return LL1, LL2, LL3, NosNod, NosLL2, NosLL3

    # Objective function
    def OF_rule(self, m):
        return sum(m.vSoC[1, 0, 0]-m.vSoC[3, 0, xv] for xv in m.sVec)

    # SoC initialisation conditiona
    def ZSoC_rule(self, m, x1, xv):
        return m.vSoC[0, x1, xv] == 0

    # Balance at different time levels
    def SoCBalance_rule(self, m, xL1, xv):
        return (m.vSoC[xL1, 0, xv] ==
                m.vSoC[m.LLTS1[xL1, 0], m.LLTS1[xL1, 1], xv] +
                m.WInFull[xL1, xv] - m.WOutFull[xL1, xv])

    # Aggregating (deterministic case)
    def SoCAggregate_rule(self, m, xL2, xv):
        return (m.vSoC[m.LLTS2[xL2, 0], 1, xv] ==
                m.vSoC[m.LLTS2[xL2, 1], m.LLTS2[xL2, 2], xv] *
                m.WghtFull[m.LLTS2[xL2, 0]] +
                m.vSoC[m.LLTS1[m.LLTS2[xL2, 0], 0],
                       m.LLTS1[m.LLTS2[xL2, 0], 1], xv] *
                (1-m.WghtFull[m.LLTS2[xL2, 0]]))

    # Aggregating (stochastic case)
    def SoCStochastic_rule(self, m, xL3, xv):
        return (m.vSoC[m.LLTS3[xL3, 0], 1, xv] ==
                m.vSoC[m.LLTS1[m.LLTS3[xL3, 0], 0],
                       m.LLTS1[m.LLTS3[xL3, 0], 1], xv] *
                (1-m.WghtFull[m.LLTS3[xL3, 0]]) +
                m.WghtFull[m.LLTS3[xL3, 0]] *
                (m.vSoC[m.LLTS3[xL3, 0], 0, xv] * -m.LLTS3[xL3, 2] +
                 sum(m.vSoC[m.LLTS3[xL3, 1]+x1, 1, xv]
                     for x1 in range(m.LLTS3[xL3, 2]+1))))

    # Initialise externally
    def initialise(self, FileName):
        # Read input data
        if self.fRea:
            self.Input_Data = self.Read(FileName)

        # Measure the size of the data arrays
        self._Measure()

        # Summarize the data as inputs, outputs, weights and uncertainty
        (WIn, WOut, Wght,
         Unc) = self.Process(self.size['Scenarios'], self.size['SumPeriods'])

        # Produce connectivity matrices
        (self.LL_TimeNodeTree, LL_TimeScenarioTree, LL_Unc, LL_WIO,
         LL_TimeNodeBeforeSequence, LL_TimeLevel,
         self.NodeTime) = self.Connect(self.size['LenPeriods'], self.size['Scenarios'], Unc, self.size['Nodes'])

        # Define inputs, outpurs and weights per node
        (self.WInFull, self.WOutFull,
         self.WghtFull) = self.Parameters(self.size['Nodes'], WIn, WOut, Wght, LL_WIO)

        (self.LL1, self.LL2, self.LL3, self.NosNod, self.NosLL2,
         self.NosLL3) = self.Link(Unc, LL_Unc, self.size['Nodes'],
                                  LL_TimeNodeBeforeSequence)

        # Weignts of the last time period (to be used by pyene)
        aux = self.LL2[LL_TimeLevel[self.size['Periods']]-1, 0]
        aux = [aux, aux+self.size['LenPeriods'][self.size['Periods']-1]]
        self.OFpyene = self.WghtFull[aux[0]:aux[1]]

        self.tree = {
                'Before': LL_TimeNodeBeforeSequence,  # Node before
                'After': self.LL_TimeNodeTree,  # Node after
                'Scenarios': LL_TimeScenarioTree,  # Scenarios
                'Uncertainty': LL_Unc,  # Flag for uncertainty
                'InOut': LL_WIO,  # Location of inputs/outputs
                'Time': self.NodeTime,  # Nodes in each time level
                }

    # Print results
    def print(self, mod):
        for xv in mod.sVec:
            print('Vector No:', xv)
            for x1 in mod.sNodz:
                print("SoC[%3.0f" % x1, "][0:1]=[%10.2f"
                      % mod.vSoC[x1, 0, xv].value, ", %10.2f"
                      % mod.vSoC[x1, 1, xv].value, "]")

    #                                   Sets                                  #
    def getSets(self, m):
        m = ConcreteModel()
        m.sNod = range(1, self.NosNod+1)
        m.sNodz = range(self.NosNod+1)
        m.sLLTS2 = range(self.NosLL2+1)
        m.sLLTS3 = range(self.NosLL3+1)
        m.FUnc = self.NosLL3
        m.sVec = range(self.size['Vectors'])

        return m

    #                                Parameters                               #
    def getPar(self, m):
        m.LLTS1 = self.LL1
        m.LLTS2 = self.LL2
        m.LLTS3 = self.LL3
        m.WInFull = self.WInFull
        m.WOutFull = self.WOutFull
        m.WghtFull = self.WghtFull

        return m

    #                             Model Variables                             #
    def getVars(self, m):
        m.vSoC= Var(m.sNodz, range(2), m.sVec, domain=NonNegativeReals,
                    initialize=0.0)

        return m

    #                               Constraints                               #
    def addCon(self, m):
        # Initialisation conditions
        m.ZSoC = Constraint(range(2), m.sVec, rule=self.ZSoC_rule)
        # Balance at different time levels
        m.SoCBalance = Constraint(m.sNodz, m.sVec, rule=self.SoCBalance_rule)
        # Aggregating (deterministic case)
        m.SOCAggregate = Constraint(m.sLLTS2, m.sVec,
                                    rule=self.SoCAggregate_rule)
        # Aggregating (stochastic case)
        if m.FUnc != 0:
            m.SoCStochastic = Constraint(m.sLLTS3, m.sVec,
                                         rule=self.SoCStochastic_rule)
        return m
