# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:04:58 2018

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
from __future__ import division
from pyomo.environ import *
import numpy as np
import networkx as nx
import json
import os

class ENetworkClass:
    # Initialize
    def __init__(self):
        # Default security considerations
        self.Sec = [2, 3]
        # Default consideration of losses
        self.Add_Loss = True
        # Defaulr demand time series
        self.DemProf = [1, 1.05, 1.1]
        # Consideration of feasibility constraints
        self.Add_Fea = True
        # Default hydropower
        self.NoHyd = 0  # 3
        self.PosHyd = [0]  # [1, 2, 3]
        self.HydPMax = [0]  # [1000, 1000, 1000]
        self.HydQMax = [0]  # [1000, 1000, 1000]
        self.HydCst = [0]  # [0, 0, 0]
        # Default intermittent renewables
        self.NoGRes = 0  # 2
        self.PosGRes = [0]  # [1, 2]
        self.GResCst = [0]  # [0, 0]
        # Default List of copies to be made
        self.h = range(1)
        # Default shift factor for vFlow_EPower
        self.hFE = [0]
        # Default shift factor for vVoltage_Angle
        self.hVA = [0]
        # Default shift factor for vEPower_Loss
        self.hEL = [0]
        # Default shift factor for vFeaB
        self.hFB = [0]
        # Shift factor for vFeaN
        self.hFN = [0]
        # Default shift factor for vGen
        self.hG = [0]
        # Default shift factor for vGenvGCost
        self.hGC = [0]
        # Default shift factor for feaibility constraints
        self.hFea = [0]
        # Add dynamic loads (e.g.,to model pumps)
        self.hDL = [0]
        self.NoDL = 0  # 2
        self.PosDL = [0]  # [2, 3]
        self.ValDL = [0]  # [0.001, 0.001]
        self.MaxDL = [0]  # [1.5, 1.5]

    # Read input data
    def Read(self, FileName):
        # Load file
        MODEL_JSON = os.path.join(os.path.dirname(__file__), '..\json',
                                  FileName)
        mpc = json.load(open(MODEL_JSON))
        ENet = nx.Graph()

        # Adding network attributes
        auxtxt = ['version', 'baseMVA', 'NoGen', 'Slack']
        for x1 in range(4):
            ENet.graph[auxtxt[x1]] = mpc[auxtxt[x1]]

        # Adding buses (nodes) and attributes
        auxtxt = ['BUS_TYPE', 'GS', 'BS', 'BUS_AREA', 'VM', 'VA', 'BASE_KV',
                  'ZONE', 'VMAX', 'VMIN']
        for xen in range(mpc["NoBus"]):
            ENet.add_node(mpc['bus']['BUS_I'][xen])
            for x1 in range(10):
                ENet.node[xen+1][auxtxt[x1]] = mpc["bus"][auxtxt[x1]][xen]

        # Adding branches (edges) and attributes
        auxtxt = ['BR_R', 'BR_X', 'BR_B', 'RATE_A', 'RATE_B', 'RATE_C', 'TAP',
                  'SHIFT', 'BR_STATUS', 'ANGMIN', 'ANGMAX']
        for xeb in range(mpc["NoBranch"]):
            xaux = [mpc["branch"]["F_BUS"][xeb], mpc["branch"]["T_BUS"][xeb]]
            ENet.add_edge(xaux[0], xaux[1])
            for x1 in range(11):
                (ENet[xaux[0]][xaux[1]]
                 [auxtxt[x1]]) = mpc["branch"][auxtxt[x1]][xeb]

        ENetDem = {
                'PD': np.array(mpc['bus']['PD'], dtype=float),
                'QD': np.array(mpc['bus']['QD'], dtype=float),
                }

        # Gen generation nodes (to mesure it)
        GenNCost = np.array(mpc['gencost']['COST'], dtype=int)
        NoOGen = len(GenNCost)
        NoCst = len(GenNCost[0])
        NoGen = NoOGen+self.NoHyd+self.NoGRes

        # Defime generation data
        ENetGen = {
                'GEN_BUS': np.zeros(NoGen, dtype=int),
                'PG': np.zeros(NoGen, dtype=float),
                'QG': np.zeros(NoGen, dtype=float),
                'QMAX': np.zeros(NoGen, dtype=float),
                'QMIN': np.zeros(NoGen, dtype=float),
                'VG': np.zeros(NoGen, dtype=float),
                'MBASE': np.zeros(NoGen, dtype=float),
                'GEN': np.zeros(NoGen, dtype=int),
                'PMAX': np.zeros(NoGen, dtype=float),
                'PMIN': np.zeros(NoGen, dtype=float),
                'PC1': np.zeros(NoGen, dtype=float),
                'PC2': np.zeros(NoGen, dtype=float),
                'QC1MIN': np.zeros(NoGen, dtype=float),
                'QC1MAX': np.zeros(NoGen, dtype=float),
                'QC2MIN': np.zeros(NoGen, dtype=float),
                'QC2MAX': np.zeros(NoGen, dtype=float),
                'RAMP_AGC': np.zeros(NoGen, dtype=float),
                'RAMP_10': np.zeros(NoGen, dtype=float),
                'RAMP_30': np.zeros(NoGen, dtype=float),
                'RAMP_Q': np.zeros(NoGen, dtype=float),
                'APF': np.zeros(NoGen, dtype=float)
                }
        ENetCost = {
                'MODEL': np.zeros(NoGen, dtype=int),
                'STARTUP': np.zeros(NoGen, dtype=float),
                'SHUTDOWN': np.zeros(NoGen, dtype=float),
                'NCOST': np.zeros(NoGen, dtype=float),
                'COST': np.zeros((NoGen, NoCst), dtype=float)
                }

        ENetGen['GEN_BUS'][0:NoOGen] = mpc['gen']['GEN_BUS']
        ENetGen['PG'][0:NoOGen] = mpc['gen']['PG']
        ENetGen['QG'][0:NoOGen] = mpc['gen']['QG']
        ENetGen['QMAX'][0:NoOGen] = mpc['gen']['QMAX']
        ENetGen['QMIN'][0:NoOGen] = mpc['gen']['QMIN']
        ENetGen['VG'][0:NoOGen] = mpc['gen']['VG']
        ENetGen['MBASE'][0:NoOGen] = mpc['gen']['MBASE']
        ENetGen['GEN'][0:NoOGen] = mpc['gen']['GEN']
        ENetGen['PMAX'][0:NoOGen] = mpc['gen']['PMAX']
        ENetGen['PMIN'][0:NoOGen] = mpc['gen']['PMIN']
        ENetGen['PC1'][0:NoOGen] = mpc['gen']['PC1']
        ENetGen['PC2'][0:NoOGen] = mpc['gen']['PC2']
        ENetGen['QC1MIN'][0:NoOGen] = mpc['gen']['QC1MIN']
        ENetGen['QC1MAX'][0:NoOGen] = mpc['gen']['QC1MAX']
        ENetGen['QC2MIN'][0:NoOGen] = mpc['gen']['QC2MIN']
        ENetGen['QC2MAX'][0:NoOGen] = mpc['gen']['QC2MAX']
        ENetGen['RAMP_AGC'][0:NoOGen] = mpc['gen']['RAMP_AGC']
        ENetGen['RAMP_10'][0:NoOGen] = mpc['gen']['RAMP_10']
        ENetGen['RAMP_30'][0:NoOGen] = mpc['gen']['RAMP_30']
        ENetGen['RAMP_Q'][0:NoOGen] = mpc['gen']['RAMP_Q']
        ENetGen['APF'][0:NoOGen] = mpc['gen']['APF']

        ENetCost['MODEL'][0:NoOGen] = mpc['gencost']['MODEL']
        ENetCost['STARTUP'][0:NoOGen] = mpc['gencost']['STARTUP']
        ENetCost['SHUTDOWN'][0:NoOGen] = mpc['gencost']['SHUTDOWN']
        ENetCost['NCOST'][0:NoOGen] = mpc['gencost']['NCOST']
        ENetCost['COST'][0:NoOGen][:] = mpc['gencost']['COST']

        # Add renewable generation
        if self.NoHyd > 0:
            # Adding hydro
            aux1 = NoOGen
            aux2 = NoOGen+self.NoHyd
            ENetGen['GEN_BUS'][aux1:aux2] = self.PosHyd
            ENetGen['PMAX'][aux1:aux2] = self.HydPMax
            ENetGen['QMAX'][aux1:aux2] = self.HydQMax
            xg2 = -1
            for xg in range(aux1, aux2):
                xg2 += 1
                ENetGen['MBASE'][xg] = ENet.graph['baseMVA']
                # Add polinomial (linear) cost curve
                if self.HydCst[xg2] != 0:
                    ENetCost['MODEL'][xg] = 2
                    ENetCost['NCOST'][xg] = 2
                    ENetCost['COST'][xg][0] = self.HydCst[xg2]

        if self.NoGRes > 0:
            # Adding intermittent generation
            aux1 = NoOGen+self.NoHyd
            aux2 = NoGen
            ENetGen['GEN_BUS'][aux1:aux2] = self.PosGRes
            xg2 = -1
            for xg in range(aux1, aux2):
                xg2 += 1
                ENetGen['PMAX'][xg] = 1000000
                ENetGen['QMAX'][xg] = 1000000
                ENetGen['MBASE'][xg] = ENet.graph['baseMVA']
                # Add polinomial (linear) cost curve
                if self.GResCst[xg2] != 0:
                    ENetCost['MODEL'][xg] = 2
                    ENetCost['NCOST'][xg] = 2
                    ENetCost['COST'][xg][0] = self.GResCst[xg2]

        return ENet, ENetDem, ENetGen, ENetCost, NoOGen, NoGen

    # Process information for optimisation purposes
    def ProcessENet(self):
        # Map connections between nodes and branches (non-sequential search)
        NoN2B = self.ENet.number_of_edges()*2+1  # Number of data points
        LLaux = np.zeros(NoN2B, dtype=int)  # connections (non-sequential)
        LLnext = np.zeros(NoN2B, dtype=int)  # Next connection (non-sequential)
        LLN2B1 = np.zeros(NoN2B, dtype=int)  # connections (sequential)
        # Position of first connection and number of cinnections
        LLN2B2 = np.zeros((self.ENet.number_of_nodes(), 4), dtype=int)

        x0 = 0  # Initial position (LLaux)
        x1 = 0  # Initial position (branches)
        for (xf, xt) in self.ENet.edges:
            x1 += 1
            auxNode = [xf-1, xt-1]
            auxX = [3, 1]
            for x2 in range(2):
                x0 += 1
                # Get next position
                xpos = LLN2B2[auxNode[x2]][auxX[x2]]
                # Initialize if the position is available
                if xpos == 0:
                    LLN2B2[auxNode[x2]][auxX[x2]] = x0
                else:  # Search for next available position
                    while LLnext[xpos] != 0:
                        xpos = LLnext[xpos]
                    # Storing data position
                    LLnext[xpos] = x0
                    (LLN2B2[auxNode[x2]]
                           [auxX[x2]-1]) = LLN2B2[auxNode[x2]][auxX[x2]-1]+1
                # Storing data point
                LLaux[x0] = x1

        # Remove the 'next' by arranging the data sequentially
        x0 = 0  # Position LLN2B1
        xacu = 1  # Total number of positions addressed so far
        for x2 in [2, 0]:
            for xn in range(self.ENet.number_of_nodes()):
                # Get first branch position for this node
                xpos = LLN2B2[xn][x2+1]
                if xpos != 0:
                    # Get other positions is available
                    LLN2B2[xn][x2+1] = xacu
                    xacu += LLN2B2[xn][x2]+1
                    for x3 in range(LLN2B2[xn][x2]+1):
                        # Store data sequentially
                        x0 = x0+1
                        LLN2B1[x0] = LLaux[xpos]
                        xpos = LLnext[xpos]

        # Set line limits
        branchNo = np.zeros((self.ENet.number_of_edges(), 2), dtype=int)
        branchData = np.zeros((self.ENet.number_of_edges(), 4), dtype=float)
        xb = 0
        for (xf, xt) in self.ENet.edges:
            branchNo[xb, :] = [xf-1, xt-1]
            branchData[xb, :4] = [self.ENet[xf][xt]['BR_R'],
                                  self.ENet[xf][xt]['BR_X'],
                                  self.ENet[xf][xt]['BR_B'],
                                  self.ENet[xf][xt]['RATE_A'] /
                                  self.ENet.graph['baseMVA']]
            xb += 1

        # Add security considerations
        NoSec2 = len(self.Sec)  # number of scenarios for security checks
        # Number of parameters required for simulating security
        NoSec1 = self.ENet.number_of_edges()+(self.ENet.number_of_edges() -
                                              1)*NoSec2
        # Auxiliaries for modelling security considerations
        # Position of the variables
        LLESec1 = np.zeros((NoSec1, 2), dtype=int)
        # Connection between the branch number and the position of the data
        LLESec2 = np.zeros((self.ENet.number_of_edges()+1, NoSec2+1),
                           dtype=int)

        for xb in range(self.ENet.number_of_edges()):
            LLESec1[xb][0] = xb
            LLESec2[xb][0] = xb
        LLESec2[self.ENet.number_of_edges()][0] = self.ENet.number_of_edges()
        x0 = self.ENet.number_of_edges()
        xacu = 0
        for xs in range(NoSec2):
            xacu += self.ENet.number_of_nodes()
            for xb in range(self.ENet.number_of_edges()):
                if xb+1 != self.Sec[xs]:
                    LLESec2[xb+1][xs+1] = x0+1
                    LLESec1[x0][:] = [xb, xacu]
                    x0 += 1

        # Add power losses estimation
        if self.Add_Loss:
            # Auxiliar for the cuadratic function to model losses
            # Choosing points for the lines
            aux = [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6,
                   2, 2.5, 3]

            # Number of points to model
            Number_LossCon = len(aux)-1

            # The function is modelled between the points a<b) as:
            # a^2-a*(b^2-a^2)/(b-a) + x *(b^2-a^2)/(b-a)
            Loss_Con1 = np.zeros(Number_LossCon, dtype=float)
            Loss_Con2 = np.zeros(Number_LossCon, dtype=float)
            for x1 in range(Number_LossCon):
                Loss_Con2[x1] = (aux[x1+1]**2-aux[x1]**2)/(aux[x1+1]-aux[x1])
                Loss_Con1[x1] = aux[x1]**2-aux[x1]*Loss_Con2[x1]
        else:
            Number_LossCon = 1
            Loss_Con1 = 0
            Loss_Con2 = 0

        # Add LL for dynamic loads
        LLDL = np.zeros(self.ENet.number_of_nodes(), dtype=int)
        for xdl in range(self.NoDL):
            LLDL[self.PosDL[xdl]-1] = xdl+1

        # Add LL for feasibility constraints (Nodes)
        LLFea = np.zeros(self.ENet.number_of_nodes()+1, dtype=int)
        if self.Add_Fea:
            NoFea = self.ENet.number_of_nodes()+1
            for xn in range(1, NoFea):
                LLFea[xn] = xn
        else:
            NoFea = 1

        return (Number_LossCon, branchNo, branchData, NoN2B, LLN2B1,
                LLN2B2, LLESec1, LLESec2, NoSec1, NoSec2, Loss_Con1,
                Loss_Con2, LLDL, NoFea, LLFea)

    # Process demand and generation parameters
    def ProcessEDem(self, ENetDem):
        # Set demand profile
        Number_Time = len(self.DemProf)
        busData = np.zeros((self.ENet.number_of_nodes(), Number_Time),
                           dtype=float)
        for xn in range(self.ENet.number_of_nodes()):
            busData[xn][0] = ENetDem['PD'][xn]/self.ENet.graph['baseMVA']
            for xt in range(Number_Time):
                busData[xn][xt] = busData[xn][0]*self.DemProf[xt]

        return Number_Time, busData

    # Process generator parameters
    def ProcessEGen(self, ENetGen, ENetCost):
        GenMax = ENetGen['PMAX']
        GenMin = ENetGen['PMIN']

        NoGenC = int(sum(ENetCost['NCOST']))
        LLGen1 = np.zeros(self.ENet.number_of_nodes(), dtype=int)
        LLGen2 = np.zeros(NoGenC, dtype=int)
        GenLCst = np.zeros((NoGenC, 2), dtype=float)
        acu = 0
        for xg in range(self.NoGen):
            # Add location of the generator
            LLGen1[ENetGen['GEN_BUS'][xg]-1] = xg+1
            # Number of cost parameters
            auxNo = int(ENetCost['NCOST'][xg])

            # Check cost function
            if ENetCost['MODEL'][xg] == 1:  # Piece-wise model
                # Collect parameters
                xval = np.zeros(auxNo, dtype=float)
                yval = np.zeros(auxNo, dtype=float)
                xc = 0
                xv = 0
                while xc <= auxNo*2-1:
                    xval[xv] = ENetCost['COST'][xg][xc]
                    yval[xv] = ENetCost['COST'][xg][xc+1]
                    xv += 1
                    xc += 2
                auxNo -= 1
            elif ENetCost['MODEL'][xg] == 2:  # Polynomial model
                # Get costs function
                fc = ENetCost['COST'][xg][:]
                xval = np.zeros(auxNo+1, dtype=float)
                yval = np.zeros(auxNo+1, dtype=float)
                # Solve equation to get parameters
                Dtx = (ENetGen['PMAX'][xg]-ENetGen['PMIN'][xg])/auxNo
                aux = ENetGen['PMIN'][xg] - Dtx
                for xv in range(auxNo+1):
                    xval[xv] = aux + Dtx
                    aux = xval[xv]
                    yval[xv] = fc[auxNo-1]
                    for xc in range(auxNo):
                        yval[xv] += fc[xc]*xval[xv]**(auxNo-xc-1)
            # Convert parameters to LP constraints
            for x1 in range(acu, acu+auxNo):
                LLGen2[x1] = xg
            for xv in range(auxNo):
                GenLCst[acu+xv][0] = (yval[xv+1] -
                                      yval[xv]) / (xval[xv+1]-xval[xv])
                GenLCst[acu+xv][1] = yval[xv]-xval[xv]*GenLCst[acu+xv][0]
            acu += auxNo

        # Changing to pu
        for xg in range(self.NoGen):
            GenMax[xg] /= self.ENet.graph['baseMVA']
            GenMin[xg] /= self.ENet.graph['baseMVA']
        for xc in range(NoGenC):
            GenLCst[xc][0] *= self.ENet.graph['baseMVA']

        return (GenMax, GenMin, LLGen1, LLGen2, NoGenC, GenLCst)

    # Print results
    def print(self, mod):
        for xh in self.h:
            print("\nCASE:", xh)

            print("\nFlow_EGen=[")
            for xn in range(1, self.NoGen+1):
                for x2 in mod.sTim:
                    aux = (mod.vGen[self.hG[xh]+xn, x2].value *
                           self.ENet.graph['baseMVA'])
                    print("%8.4f " % aux, end='')
                print()
            print("];")

            print("\nFlow_EPower=[")
            for x1 in range(1, self.ENet.number_of_edges()+1):
                for x2 in mod.sTim:
                    aux = (mod.vFlow_EPower[self.hFE[xh]+x1, x2].value *
                           self.ENet.graph['baseMVA'])
                    print("%8.4f " % aux, end='')
                print()
            print("];")

            # Voltage angles
            print("\nVoltage_Angle=[")
            for xn in mod.sBuses:
                for xt in mod.sTim:
                    aux = mod.vVoltage_Angle[self.hVA[xh]+xn, xt].value
                    print("%8.4f " % aux, end='')
                print()
            print("];")

            # Power losses
            print("\nEPower_Loss=[")
            for xb in range(1, self.ENet.number_of_edges()+1):
                for xt in mod.sTim:
                    aux = mod.vEPower_Loss[self.hEL[xh]+xb, xt].value
                    print("%8.4f " % aux, end='')
                print()
            print("];")

            # Dinamic loads
            print("\nvGenDL=[")
            for xdl in mod.sDL:
                for xt in mod.sTim:
                    aux = mod.vGenDL[self.hDL[xh]+xdl, xt].value
                    print("%8.4f " % aux, end='')
                print()
            print("];")

            # Feasibility constraints
            print("\nFeas=[")
            for xf in mod.sFea:
                for xt in mod.sTim:
                    aux = mod.vFea[self.hFea[xh]+xf, xt].value
                    print("%8.4f " % aux, end='')
                print()
            print("];")

    # Initialize externally
    def initialise(self, FileName):
        # Read network data
        (self.ENet, ENetDem, ENetGen, ENetCost, self.NoOGen,
         self.NoGen) = self.Read(FileName)

        (self.Number_LossCon, self.branchNo, self.branchData, self.NoN2B,
         self.LLN2B1, self.LLN2B2, self.LLESec1, self.LLESec2,
         self.NoSec1, self.NoSec2, self.Loss_Con1, self.Loss_Con2, self.LLDL,
         self.NoFea, self.LLFea) = self.ProcessENet()

        (self.Number_Time,
         self.busData) = self.ProcessEDem(ENetDem)

        (self.GenMax, self.GenMin, self.LLGen1, self.LLGen2, self.NoGenC,
         self.GenLCst) = self.ProcessEGen(ENetGen, ENetCost)

        self.NoBuses = self.ENet.number_of_nodes()*(1+self.NoSec2)-1
        self.NoBranch = (self.ENet.number_of_edges() +
                         (self.ENet.number_of_edges()-1)*self.NoSec2)

    # Objective function
    def OF_rule(self, m):
        xh = self.h[0]
        return (sum(sum(m.vGCost[self.hGC[xh]+xg, xt] for xg in m.sGen) +
                    sum(m.vFea[self.hFea[xh]+xf, xt] for xf in m.sFea) *
                    1000000 for xt in m.sTim) -
                sum(m.ValDL[xdl]*sum(m.vGenDL[self.hDL[xh]+xdl+1, xt]
                                     for xt in m.sTim) for xdl in m.sDL))

    # Reference line flow
    def EPow0_rule(self, m, xt, xh):
        return m.vFlow_EPower[self.hFE[xh], xt] == 0

    # Reference generation
    def EGen0_rule(self, m, xt, xh):
        return m.vGen[self.hG[xh], xt] == 0

    # Maximum generation
    def EGMax_rule(self, m, xg, xt, xh):
        return m.vGen[self.hG[xh]+xg+1, xt] <= m.GenMax[xg]

    # Minimum generation
    def EGMin_rule(self, m, xg, xt, xh):
        return m.vGen[self.hG[xh]+xg+1, xt] >= m.GenMin[xg]

    # Piece-wise generation costs approximation
    def EGenC_rule(self, m, xc, xt, xh):
        return (m.vGCost[self.hGC[xh]+m.LLGen2[xc], xt] >=
                m.vGen[self.hG[xh]+m.LLGen2[xc]+1, xt] *
                m.GenLCst[xc, 0]+m.GenLCst[xc, 1])

    # Branch flows
    def EFlow_rule(self, m, xt, xb, xh):
        xaux1 = self.hVA[xh]+m.branchNo[m.LLESec1[xb, 0], 0]+m.LLESec1[xb, 1]
        xaux2 = self.hVA[xh]+m.branchNo[m.LLESec1[xb, 0], 1]+m.LLESec1[xb, 1]
        return (m.vFlow_EPower[self.hFE[xh]+xb+1, xt] ==
                (m.vVoltage_Angle[xaux1, xt]-m.vVoltage_Angle[xaux2, xt]) /
                m.branchData[m.LLESec1[xb, 0], 1])

    # Branch capacity constraint (positive)
    def EFMax_rule(self, m, xt, xb, xh):
        return (m.vFlow_EPower[self.hFE[xh]+xb+1, xt] >=
                -m.branchData[m.LLESec1[xb, 0], 3])

    # Branch capacity constraint (negative)
    def EFMin_rule(self, m, xt, xb, xh):
        return (m.vFlow_EPower[self.hFE[xh]+xb+1, xt] <=
                m.branchData[m.LLESec1[xb, 0], 3])

    # Nodal balance: Generation + Flow in - loss/2 = Demand + flow out + loss/2
    def EBalance_rule(self, m, xn, xt, xs, xh):
        return (m.vGen[self.hG[xh]+m.LLGen1[xn], xt] +
                sum(m.vFlow_EPower[self.hFE[xh] +
                                   m.LLESec2[m.LLN2B1[x2+m.LLN2B2[xn, 1]], xs],
                                   xt] -
                    m.vEPower_Loss[self.hEL[xh] +
                                   m.LLN2B1[x2+m.LLN2B2[xn, 1]], xt]/2
                    for x2 in range(1+m.LLN2B2[xn, 0])) ==  
                m.busData[xn, xt]-m.vFea[self.hFea[xh]+m.LLFea[xn], xt] +
                m.vGenDL[self.hDL[xh]+m.LLDL[xn], xt] +
                sum(m.vFlow_EPower[self.hFE[xh] +
                                   m.LLESec2[m.LLN2B1[x1+m.LLN2B2[xn, 3]], xs],
                                   xt] +
                    m.vEPower_Loss[self.hEL[xh] +
                                   m.LLN2B1[x1+m.LLN2B2[xn, 3]], xt]/2
                    for x1 in range(1+m.LLN2B2[xn, 2])))

    # Power losses (Positive)
    def DCLossA_rule(self, m, xb, xb2, xt, xh):
        return (m.vEPower_Loss[self.hEL[xh]+xb+1, xt] >=
                (m.Loss_Con1[xb2]+m.vFlow_EPower[self.hFE[xh]+xb+1, xt] *
                 m.Loss_Con2[xb2])*m.branchData[xb, 0])

    # Power losses (Negative)
    def DCLossB_rule(self, m, xb, xb2, xt, xh):
        return (m.vEPower_Loss[self.hEL[xh]+xb+1, xt] >=
                (m.Loss_Con1[xb2] -
                 m.vFlow_EPower[self.hFE[xh]+xb+1, xt]*m.Loss_Con2[xb2]) *
                m.branchData[xb, 0])

    # No losses
    def DCLossN_rule(self, m, xb, xt, xh):
        return m.vEPower_Loss[self.hEL[xh]+xb+1, xt] == 0

    # Maximum capacity of dynamic loads
    def LDMax_rule(self, m, xdl, xt, xh):
        return m.vGenDL[self.hDL[xh]+xdl, xt] <= m.MaxDL[xdl]

    # Initialising dynamic loads
    def LDIni_rule(self, m, xt, xh):
        return m.vGenDL[self.hDL[xh], xt] == 0

    # Positions without feasibility constraints
    def setFea_rule(self, m, xt, xh):
        return m.vFea[self.hFea[xh], xt] == 0

    #                                   Sets                                  #
    def getSets(self, m):
        m.sBra = range(self.ENet.number_of_edges())
        m.sBraP = range(self.ENet.number_of_edges()+1)
        m.sBus = range(self.ENet.number_of_nodes())
        m.sTim = range(self.Number_Time)
        m.sLoss = range(self.Number_LossCon)
        m.sBranch = range(self.NoBranch+1)
        m.sBuses = range(self.NoBuses+1)
        m.sN2B = range(self.NoN2B)
        m.sSec1 = range(self.NoSec1)
        m.sSec2 = range(self.NoSec2+1)
        m.sGen = range(self.NoGen)
        m.sGenP = range(self.NoGen+1)
        m.sGenC = range(self.NoGenC+1)
        m.sGenCM = range(self.NoGenC)
        m.sDL = range(self.NoDL)
        m.sFea = range(self.NoFea)

        return m

    #                                Parameters                               #
    def getPar(self, m):
        m.branchNo = self.branchNo
        m.branchData = self.branchData
        m.busData = self.busData
        m.Loss_Con1 = self.Loss_Con1
        m.Loss_Con2 = self.Loss_Con2
        m.LLN2B1 = self.LLN2B1
        m.LLN2B2 = self.LLN2B2
        m.LLESec1 = self.LLESec1
        m.LLESec2 = self.LLESec2
        m.GenMax = self.GenMax
        m.GenMin = self.GenMin
        m.LLGen1 = self.LLGen1
        m.LLGen2 = self.LLGen2
        m.GenLCst = self.GenLCst
        m.ValDL = self.ValDL
        m.MaxDL = self.MaxDL
        m.LLDL = self.LLDL
        m.LLFea = self.LLFea

        return m

    #                             Model Variables                             #
    def getVars(self, m):
        Noh = len(self.h)
        m.vFlow_EPower = Var(range(Noh*(self.NoBranch+1)), m.sTim,
                             domain=Reals, initialize=0.0)
        m.vVoltage_Angle = Var(range(Noh*(self.NoBuses+1)), m.sTim,
                               domain=Reals, initialize=0.0)
        m.vEPower_Loss = Var(range(Noh*(self.ENet.number_of_edges()+1)),
                             m.sTim, domain=NonNegativeReals, initialize=0.0)
        m.vFea = Var(range(Noh*self.NoFea), m.sTim, domain=NonNegativeReals,
                     initialize=0.0)
        m.vGen = Var(range(Noh*(self.NoGen+1)), m.sTim, domain=NonNegativeReals,
                     initialize=0.0)
        m.vGCost = Var(range(Noh*self.NoGen), m.sTim, domain=NonNegativeReals,
                       initialize=0.0)
        m.vGenDL = Var(range(Noh*(self.NoDL+1)), m.sTim,
                       domain=NonNegativeReals, initialize=0.0)

        return m

    #                               Constraints                               #
    def addCon(self, m):
        # Reference line flow
        m.EPow0 = Constraint(m.sTim, self.h, rule=self.EPow0_rule)
        # Reference generation
        m.EGen0 = Constraint(m.sTim, self.h, rule=self.EGen0_rule)
        # Maximum generation
        m.EGMax = Constraint(m.sGen, m.sTim, self.h, rule=self.EGMax_rule)
        # Minimum generation
        m.EGMin = Constraint(m.sGen, m.sTim, self.h, rule=self.EGMin_rule)
        # Piece-wise generation costs approximation
        m.EGenC = Constraint(m.sGenCM, m.sTim, self.h, rule=self.EGenC_rule)
        # Branch flows
        m.EFlow = Constraint(m.sTim, m.sSec1, self.h, rule=self.EFlow_rule)
        # Branch capacity constraint (positive)
        m.EFMax = Constraint(m.sTim, m.sSec1, self.h, rule=self.EFMax_rule)
        # Branch capacity constraint (negative)
        m.EFMin = Constraint(m.sTim, m.sSec1, self.h, rule=self.EFMin_rule)
        # Balance: Generation + Flow in - loss/2 = Demand + flow out + loss/2
        m.EBalance = Constraint(m.sBus, m.sTim, m.sSec2, self.h,
                                rule=self.EBalance_rule)
        # Dinamic load maximum capacity
        m.DLMax = Constraint(m.sDL, m.sTim, self.h, rule=self.LDMax_rule)
        # Dinamic load initialisation
        m.DLIni = Constraint(m.sTim, self.h, rule=self.LDIni_rule)
        # Feasibility constraints
        m.setFea = Constraint(m.sTim, self.h, rule=self.setFea_rule)
        # Adding piece wise estimation of losses
        if self.Add_Loss:
            m.DCLossA = Constraint(m.sBra, m.sLoss,
                                   m.sTim, self.h, rule=self.DCLossA_rule)
            m.DCLossB = Constraint(m.sBra, m.sLoss,
                                   m.sTim, self.h, rule=self.DCLossB_rule)
        else:
            m.DCLossNo = Constraint(mod.sBra, mod.sTim, self.h,
                                    rule=self.DCLossN_rule)
        return m
