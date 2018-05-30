# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:04:58 2018

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
from __future__ import division
from pyomo.environ import *
import math
import numpy as np
import networkx as nx
import json
import os


class ENetworkClass:
    # Initialize
    def __init__(self):
        # Basic settings
        self.settings = {
                'NoTime': 1,  # Number of time steps
                'Security': [],  # Security constraints (lines)
                'Losses': True,  # Consideration of losses
                'Feasibility': True,  # Feasibility constraints
                'Pieces': 0  # Size of pieces used for piece-wise estimations
                }
        # Connections
        self.connections = {
                'set': range(1),  # Connections between nodes
                'Flow': [0],  # Power flow through the lines
                'Voltage': [0],  # Voltages in each node
                'Loss': [0],  # Power losses
                'Feasibility': [0],  # Lines to trip for security consideration
                'Generation': [0],  # Location (bus) of generators
                'Cost': [0],  # Generator consts
                'Pump': [0]
                }
        # Scenarios
        self.scenarios = {
                'Number': 1,  # Number of scenarios
                'Links': 'Default',  # Links between the buses and profiles
                'NoDem': 0,  # Number of demand profiles
                'Demand': [1],  # Demand profiles
                'NoRES': 0,  # Number of RES profiles
                'LinksRes': 'Default',  # Links RES generators and profiles
                'RES': [0]  # Location of the RES profiles
                }
        # Hydropower
        self.hydropower = {
                'Number': 0,  # Number of hydropower plants
                'Bus': [0],  # Location (Bus) in the network
                'Max': [0],  # Capacity (kW)
                'Cost': [0]  # Costs
                }
        # Pumps
        self.pumps = {
                'Number': 0,  # Number of pumps
                'Bus': [0],  # Location (Bus) in the network
                'Max': [0],  # Capacity (kW)
                'Value': [0]  # Value (OF)
                }
        # RES
        self.RES = {
                'Number': 0,  # Number of RES generators
                'Bus': [0],  # Location (Bus) in the network
                'Max': [0],  # Capacity (kW)
                'Cost': [0]  # Cost (OF)
                }
        self.Storage = {
                'Number': 0,  # Number of RES generators
                'Bus': [0],  # Location (Bus) in the network
                'Max': [0],  # Capacity (kW)
                'Efficiency': [0]  # Round trip efficiency
                }
        self.Print = {
                'Generation': True,
                'Flows': True,
                'Voltages': True,
                'Losses': True,
                'Curtailment': True,
                'Feasibility': True,
                }

    # Read input data
    def Read(self, FileName):
        # Load file
        MODEL_JSON = os.path.join(os.path.dirname(__file__), '..', 'json',
                                  FileName)
        mpc = json.load(open(MODEL_JSON))
        self.networkE = nx.Graph()

        # Adding network attributes
        aux = ['version', 'baseMVA', 'NoGen', 'Slack']
        for x1 in range(4):
            self.networkE.graph[aux[x1]] = mpc[aux[x1]]

        # Adding buses (nodes) and attributes
        aux = ['BUS_TYPE', 'GS', 'BS', 'BUS_AREA', 'VM', 'VA', 'BASE_KV',
               'ZONE', 'VMAX', 'VMIN']
        for xen in range(mpc["NoBus"]):
            self.networkE.add_node(mpc['bus']['BUS_I'][xen])
            for x1 in range(10):
                self.networkE.node[xen+1][aux[x1]] = mpc["bus"][aux[x1]][xen]

        # Adding branches (edges) and attributes
        aux = ['BR_R', 'BR_X', 'BR_B', 'RATE_A', 'RATE_B', 'RATE_C', 'TAP',
               'SHIFT', 'BR_STATUS', 'ANGMIN', 'ANGMAX']
        for xeb in range(mpc["NoBranch"]):
            xaux = [mpc["branch"]["F_BUS"][xeb], mpc["branch"]["T_BUS"][xeb]]
            self.networkE.add_edge(xaux[0], xaux[1])
            for x1 in range(11):
                (self.networkE[xaux[0]][xaux[1]]
                 [aux[x1]]) = mpc["branch"][aux[x1]][xeb]

        self.demandE = {
                'PD': np.array(mpc['bus']['PD'], dtype=float),
                'QD': np.array(mpc['bus']['QD'], dtype=float),
                }

        # Gen generation nodes (to mesure it)
        GenNCost = np.array(mpc['gencost']['COST'], dtype=int)
        NoOGen = len(GenNCost)
        NoCst = len(GenNCost[0])
        NoGen = NoOGen+self.hydropower['Number']+self.RES['Number']

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

        # Adjust demand dimensions
        self.scenarios['Demand'] = np.asarray(self.scenarios['Demand'])

        # Default settings for demand profiles
        if self.scenarios['Links'] == 'Default':
            self.scenarios['Links'] = np.ones(self.networkE.number_of_nodes() *
                                              self.scenarios['Number'],
                                              dtype=int)
            acu = self.networkE.number_of_nodes()
            for xs in range(self.scenarios['Number']-1):
                for xt in range(self.networkE.number_of_nodes()):
                    self.scenarios['Links'][acu] = xs+2
                    acu += 1

        # Default settings for RES profiles
        # i.e., each scenario is linked to a profile
        if self.scenarios['LinksRes'] == 'Default':
            self.scenarios['LinksRes'] = np.ones(self.scenarios['Number'] *
                                                 self.RES['Number'], dtype=int)
            acu = self.RES['Number']
            for xs in range(self.scenarios['Number']-1):
                for xt in range(self.RES['Number']):
                    self.scenarios['LinksRes'][acu] = xs+2
                    acu += 1

        # Add renewable generation
        if self.hydropower['Number'] > 0:
            # Adding hydro
            aux1 = NoOGen
            aux2 = NoOGen+self.hydropower['Number']
            ENetGen['GEN_BUS'][aux1:aux2] = self.hydropower['Bus']
            ENetGen['PMAX'][aux1:aux2] = self.hydropower['Max']
            xg2 = -1
            for xg in range(aux1, aux2):
                xg2 += 1
                ENetGen['MBASE'][xg] = self.networkE.graph['baseMVA']
                # Add polinomial (linear) cost curve
                if self.hydropower['Cost'][xg2] != 0:
                    ENetCost['MODEL'][xg] = 2
                    ENetCost['NCOST'][xg] = 2
                    ENetCost['COST'][xg][0] = self.hydropower['Cost'][xg2]

        if self.RES['Number'] > 0:
            # Adding intermittent generation
            aux = NoOGen+self.hydropower['Number']+1
            aux1 = NoOGen+self.hydropower['Number']
            aux2 = NoGen
            ENetGen['GEN_BUS'][aux1:aux2] = self.RES['Bus']
            xg2 = -1
            for xg in range(aux1, aux2):
                xg2 += 1
                ENetGen['PMAX'][xg] = 1000000
                ENetGen['QMAX'][xg] = 1000000
                ENetGen['MBASE'][xg] = self.networkE.graph['baseMVA']
                # Add polinomial (linear) cost curve
                if self.RES['Cost'][xg2] != 0:
                    ENetCost['MODEL'][xg] = 2
                    ENetCost['NCOST'][xg] = 2
                    ENetCost['COST'][xg][0] = self.RES['Cost'][xg2]
        self.generationE = {
                'Data': ENetGen,
                'Costs': ENetCost,
                'Number': NoGen
                }

    # Process information for optimisation purposes
    def ProcessENet(self):
        # Map connections between nodes and branches (non-sequential search)
        NoN2B = self.networkE.number_of_edges()*2+1  # Number of data points
        LLaux = np.zeros(NoN2B, dtype=int)  # connections (non-sequential)
        LLnext = np.zeros(NoN2B, dtype=int)  # Next connection (non-sequential)
        LLN2B1 = np.zeros(NoN2B, dtype=int)  # connections (sequential)
        # Position of first connection and number of cinnections
        LLN2B2 = np.zeros((self.networkE.number_of_nodes(), 4), dtype=int)

        x0 = 0  # Initial position (LLaux)
        x1 = 0  # Initial position (branches)
        for (xf, xt) in self.networkE.edges:
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
            for xn in range(self.networkE.number_of_nodes()):
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
        branchNo = np.zeros((self.networkE.number_of_edges(), 2), dtype=int)
        branchData = np.zeros((self.networkE.number_of_edges(), 4),
                              dtype=float)
        xb = 0
        for (xf, xt) in self.networkE.edges:
            branchNo[xb, :] = [xf-1, xt-1]
            branchData[xb, :4] = [self.networkE[xf][xt]['BR_R'],
                                  self.networkE[xf][xt]['BR_X'],
                                  self.networkE[xf][xt]['BR_B'],
                                  self.networkE[xf][xt]['RATE_A'] /
                                  self.networkE.graph['baseMVA']]
            xb += 1

        # Add security considerations
        NoSec2 = len(self.settings['Security'])
        # Number of parameters required for simulating security
        aux = self.networkE.number_of_edges()
        NoSec1 = self.networkE.number_of_edges()*(1+NoSec2)-NoSec2
        # Auxiliaries for modelling security considerations
        # Position of the variables
        LLESec1 = np.zeros((NoSec1, 2), dtype=int)
        # Connection between the branch number and the position of the data
        LLESec2 = np.zeros((self.networkE.number_of_edges()+1, NoSec2+1),
                           dtype=int)

        for xb in range(self.networkE.number_of_edges()):
            LLESec1[xb][0] = xb
            LLESec2[xb][0] = xb
        aux = self.networkE.number_of_edges()
        LLESec2[aux][0] = aux
        x0 = aux
        xacu = 0
        for xs in range(NoSec2):
            xacu += self.networkE.number_of_nodes()
            for xb in range(self.networkE.number_of_edges()):
                if xb+1 != self.settings['Security'][xs]:
                    LLESec2[xb+1][xs+1] = x0+1
                    LLESec1[x0][:] = [xb, xacu]
                    x0 += 1

        # Add power losses estimation
        if self.settings['Losses']:
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
                Loss_Con2[x1] = ((aux[x1+1]**2-aux[x1]**2) /
                                    (aux[x1+1]-aux[x1]))
                Loss_Con1[x1] = aux[x1]**2-aux[x1]*Loss_Con2[x1]
        else:
            Number_LossCon = 1
            Loss_Con1 = 0
            Loss_Con2 = 0

        # Add LL for dynamic loads
        LLDL = np.zeros(self.networkE.number_of_nodes(), dtype=int)
        for xdl in range(self.pumps['Number']):
            LLDL[self.pumps['Bus'][xdl]-1] = xdl+1

        # Add LL for feasibility constraints (Nodes)
        LLFea = np.zeros(self.networkE.number_of_nodes()+1, dtype=int)
        if self.settings['Feasibility']:
            NoFea = self.networkE.number_of_nodes()+1
            for xn in range(1, NoFea):
                LLFea[xn] = xn
        else:
            NoFea = 1
        self.LL = {}
        self.NoLL = {}
        self.iLL = {}

        return (Number_LossCon, branchNo, branchData, NoN2B, LLN2B1,
                LLN2B2, LLESec1, LLESec2, NoSec1, NoSec2, Loss_Con1,
                Loss_Con2, LLDL, NoFea, LLFea)

    # Process demand and generation parameters
    def ProcessEDem(self, ENetDem):
        # Adjust demand profiles
        busData = np.zeros(self.networkE.number_of_nodes(), dtype=float)
        for xn in range(self.networkE.number_of_nodes()):
            busData[xn] = self.demandE['PD'][xn]/self.networkE.graph['baseMVA']
        # Auxiliar to find demand profiles
        busScenario = np.zeros((self.networkE.number_of_nodes(),
                                self.scenarios['Number']), dtype=int)
        acu = 0
        for xs in range(self.scenarios['Number']):
            for xn in range(self.networkE.number_of_nodes()):
                busScenario[xn][xs] = ((self.scenarios['Links'][acu]-1) *
                                       self.settings['NoTime'])
                acu += 1

        # Auxiliar to find RES profiles
        resScenario = np.zeros((self.RES['Number'], self.scenarios['Number'],
                                2), dtype=int)
        for xh in range(self.scenarios['Number']):
            for xg in range(self.RES['Number']):
                # Generator location
                resScenario[xg][xh][0] = (1+xg+xh-self.RES['Number'] +
                                          self.generationE['Number']*(1+xh))
                # Profile location
                resScenario[xg][xh][1] = ((self.scenarios['LinksRes']
                                           [xg+xh*(self.RES['Number'])]-1) *
                                          self.settings['NoTime'])

        return (busData, busScenario, resScenario)

    # Produce LL while removing the 'next'
    def _getLL(self, NoLL, NoDt, DtLL):
        # Produce LL and 'next'
        LL = np.zeros(NoLL, dtype=int)
        LLnext = np.zeros(NoDt, dtype=int)
        for xd in range(NoDt):
            xpos = DtLL[xd]
            # Is it the first one?
            if LL[xpos-1] == 0:
                LL[xpos-1] = xd+1
            else:
                xpos = LL[xpos-1]
                while LLnext[xpos-1] != 0:
                    xpos = LLnext[xpos-1]
                LLnext[xpos-1] = xd+1

        # Sort data so that the 'next' can be removed
        LL1 = np.zeros(NoDt+1, dtype=int)
        LL2 = np.zeros((NoLL, 2), dtype=int)
        xL = 1
        for xn in range(NoLL):
            xpos = LL[xn]
            if xpos != 0:
                LL2[xn][0] = xL
                # Add generators
                while xpos != 0:
                    LL1[xL] = xpos
                    xpos = LLnext[xpos-1]
                    xL += 1
                LL2[xn][1] = xL-1

        return (LL1, LL2)

    # Process generator parameters
    def ProcessEGen(self):
        GenMax = self.generationE['Data']['PMAX']
        GenMin = self.generationE['Data']['PMIN']

        # Get LL for generators
        (LLGen1, LLGen2) = self._getLL(self.networkE.number_of_nodes(),
                                       self.generationE['Number'],
                                       self.generationE['Data']['GEN_BUS'])

        # Get number of reqruied piece wise cost constraints
        pwNo = np.zeros(self.generationE['Number'], dtype=int)
        NoGenC = 0
        if self.settings['Pieces'] == 0:
            for xg in range(self.generationE['Number']):
                pwNo[xg] = self.generationE['Costs']['NCOST'][xg]
                NoGenC += pwNo[xg]
        else:
            for xg in range(self.generationE['Number']):
                if self.generationE['Costs']['MODEL'][xg] == 1:
                    NoGenC += self.generationE['Costs']['NCOST'][xg]
                elif self.generationE['Costs']['MODEL'][xg] == 2:  # Pol model
                    pwNo[xg] = math.ceil((self.generationE['Data']
                                          ['PMAX'][xg] -
                                          self.generationE['Data']
                                          ['PMIN'][xg]) /
                                         self.settings['Pieces'])
                    NoGenC += pwNo[xg]

        LLGenC = np.zeros(NoGenC, dtype=int)
        GenLCst = np.zeros((NoGenC, 2), dtype=float)
        acu = 0
        for xg in range(self.generationE['Number']):
            # Number of cost parameters
            auxNo = int(self.generationE['Costs']['NCOST'][xg])

            # Check cost function
            if self.generationE['Costs']['MODEL'][xg] == 1:  # Piece-wise model
                # Collect parameters
                xval = np.zeros(auxNo, dtype=float)
                yval = np.zeros(auxNo, dtype=float)
                xc = 0
                xv = 0
                while xc <= auxNo*2-1:
                    xval[xv] = self.generationE['Costs']['COST'][xg][xc]
                    yval[xv] = self.generationE['Costs']['COST'][xg][xc+1]
                    xv += 1
                    xc += 2
                auxNo -= 1
            elif self.generationE['Costs']['MODEL'][xg] == 2:  # Pol model
                # Get costs function
                fc = self.generationE['Costs']['COST'][xg][:]
                xval = np.zeros(pwNo[xg]+1, dtype=float)
                yval = np.zeros(pwNo[xg]+1, dtype=float)

                # Solve equation to get parameters
                Dtx = (self.generationE['Data']['PMAX'][xg] -
                       self.generationE['Data']['PMIN'][xg])/pwNo[xg]
                aux = self.generationE['Data']['PMIN'][xg] - Dtx
                for xv in range(pwNo[xg]+1):
                    xval[xv] = aux + Dtx
                    aux = xval[xv]
                    yval[xv] = fc[auxNo-1]
                    for xc in range(auxNo):
                        yval[xv] += fc[xc]*xval[xv]**(auxNo-xc-1)

            # Convert parameters to LP constraints
            for x1 in range(acu, acu+pwNo[xg]):
                LLGenC[x1] = xg
            for xv in range(pwNo[xg]):
                GenLCst[acu+xv][0] = (yval[xv+1] -
                                      yval[xv]) / (xval[xv+1]-xval[xv])
                GenLCst[acu+xv][1] = yval[xv]-xval[xv]*GenLCst[acu+xv][0]
            acu += pwNo[xg]

        # Changing to pu
        for xg in range(self.generationE['Number']):
            GenMax[xg] /= self.networkE.graph['baseMVA']
            GenMin[xg] /= self.networkE.graph['baseMVA']
        for xc in range(NoGenC):
            GenLCst[xc][0] *= self.networkE.graph['baseMVA']

        return (GenMax, GenMin, LLGen1, LLGen2, LLGenC, NoGenC, GenLCst)

    # Switch off print flags
    def offPrint(self):
        self.Print['Generation'] = False
        self.Print['Flows'] = False
        self.Print['Voltages'] = False
        self.Print['Losses'] = False
        self.Print['Curtailment'] = False
        self.Print['Feasibility'] = False

        # Print results
    def print(self, mod):
        for xh in mod.sCon:
            print("\nCASE:", xh)

            if self.Print['Generation']:
                print("\nFlow_EGen=[")
                for xn in range(1, self.generationE['Number']+1):
                    for x2 in mod.sTim:
                        aux = (mod.vGen[self.connections['Generation'][xh]+xn,
                                        x2].value *
                               self.networkE.graph['baseMVA'])
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Flows']:
                print("\nFlow_EPower=[")
                for x1 in range(1, self.networkE.number_of_edges()+1):
                    for x2 in mod.sTim:
                        aux = (mod.vFlow[self.connections['Flow'][xh] +
                                         x1, x2].value *
                               self.networkE.graph['baseMVA'])
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Voltages']:
                print("\nVoltage_Angle=[")
                for xn in mod.sBuses:
                    for xt in mod.sTim:
                        aux = self.connections['Voltage'][xh]
                        aux = mod.vVolt[aux+xn, xt].value
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Losses']:
                print("\nEPower_Loss=[")
                for xb in range(1, self.networkE.number_of_edges()+1):
                    for xt in mod.sTim:
                        aux = (mod.vLoss[self.connections['Loss'][xh]+xb,
                                        xt].value *
                               self.networkE.graph['baseMVA'])
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Curtailment']:
                print("\nvDL=[")
                for xdl in mod.sDL:
                    for xt in mod.sTim:
                        aux = mod.vDL[self.connections['Pump'][xh]+xdl,
                                      xt].value
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Feasibility']:
                print("\nFeas=[")
                for xf in mod.sFea:
                    for xt in mod.sTim:
                        aux = mod.vFea[self.connections['Feasibility'][xh]+xf,
                                       xt].value
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

    # Initialize externally
    def initialise(self, FileName):
        # Read network data
        self.Read(FileName)

        (self.Number_LossCon, self.branchNo, self.branchData, self.NoN2B,
         self.LLN2B1, self.LLN2B2, self.LLESec1, self.LLESec2,
         self.NoSec1, self.NoSec2, self.Loss_Con1, self.Loss_Con2, self.LLDL,
         self.NoFea, self.LLFea) = self.ProcessENet()

        (self.busData, self.busScenario,
         self.resScenario) = self.ProcessEDem(self.demandE)

        (self.GenMax, self.GenMin, self.LLGen1, self.LLGen2, self.LLGenC,
         self.NoGenC, self.GenLCst) = self.ProcessEGen()

        self.NoBuses = self.networkE.number_of_nodes()*(1+self.NoSec2)-1
        self.NoBranch = (self.networkE.number_of_edges() +
                         (self.networkE.number_of_edges()-1)*self.NoSec2)

        self.LLStor = np.zeros((self.networkE.number_of_nodes(),
                                self.scenarios['Number']), dtype=int)
        acu = 0
        for xh in range(self.scenarios['Number']):
            for x in range(self.Storage['Number']):
                acu += 1
                self.LLStor[self.Storage['Bus'][x]-1][xh] = acu

        self.LLTime = np.zeros(self.settings['NoTime'], dtype=int)
        self.LLTime[0] = self.settings['NoTime']-1
        for xt in range(1, self.settings['NoTime']):
            self.LLTime[xt] = xt-1

    # Objective function
    def OF_rule(self, m):
        xh = self.connections['set'][0]
        return (sum(sum(m.vGCost[self.connections['Cost'][xh]+xg, xt]
                        for xg in m.sGen) +
                    sum(m.vFea[self.connections['Feasibility'][xh]+xf, xt]
                        for xf in m.sFea) *
                    1000000 for xt in m.sTim) -
                sum(m.ValDL[xdl]*sum(m.vDL[self.connections['Pump'][xh] +
                                     xdl+1, xt]
                                     for xt in m.sTim) for xdl in m.sDL))

    # Reference line flow
    def EPow0_rule(self, m, xt, xh):
        return m.vFlow[self.connections['Flow'][xh], xt] == 0

    # Reference generation
    def EGen0_rule(self, m, xt, xh):
        return m.vGen[self.connections['Generation'][xh], xt] == 0

    # Maximum generation
    def EGMax_rule(self, m, xg, xt, xh):
        return (m.vGen[self.connections['Generation'][xh]+xg+1, xt] <=
                m.GenMax[xg])

    # Maximum RES generation
    def RESMax_rule(self, m, xt, xg, xh):
        return (m.vGen[self.resScenario[xg][xh][0], xt] <=
                self.scenarios['RES'][self.resScenario[xg][xh][1]+xt] *
                self.RES['Max'][xg])

    # Minimum generation
    def EGMin_rule(self, m, xg, xt, xh):
        return (m.vGen[self.connections['Generation'][xh]+xg+1, xt] >=
                m.GenMin[xg])

    # Piece-wise generation costs approximation
    def EGenC_rule(self, m, xc, xt, xh):
        return (m.vGCost[self.connections['Cost'][xh]+m.LLGenC[xc], xt] >=
                m.vGen[self.connections['Generation'][xh]+m.LLGenC[xc]+1, xt] *
                m.GenLCst[xc, 0]+m.GenLCst[xc, 1])

    # Branch flows
    def EFlow_rule(self, m, xt, xb, xh):
        aux = self.connections['Voltage'][xh]+m.LLESec1[xb, 1]
        xaux1 = aux+m.branchNo[m.LLESec1[xb, 0], 0]
        xaux2 = aux+m.branchNo[m.LLESec1[xb, 0], 1]
        return (m.vFlow[self.connections['Flow'][xh]+xb+1, xt] ==
                (m.vVolt[xaux1, xt]-m.vVolt[xaux2, xt]) /
                m.branchData[m.LLESec1[xb, 0], 1])

    # Branch capacity constraint (positive)
    def EFMax_rule(self, m, xt, xb, xh):
        return (m.vFlow[self.connections['Flow'][xh]+xb+1, xt] >=
                -m.branchData[m.LLESec1[xb, 0], 3])

    # Branch capacity constraint (negative)
    def EFMin_rule(self, m, xt, xb, xh):
        return (m.vFlow[self.connections['Flow'][xh]+xb+1, xt] <=
                m.branchData[m.LLESec1[xb, 0], 3])

    # Nodal balance: Generation + Flow in - loss/2 = Demand + flow out + loss/2
    def EBalance_rule(self, m, xn, xt, xs, xh):
        if self.LLStor[xn, xh] == 0:
            aux = 0
        else:
            aux = self.Storage['Efficiency'][self.LLStor[xn, 0]-1]

        return (sum(m.vGen[self.connections['Generation'][xh]+m.LLGen1[xg], xt]
                    for xg in range(m.LLGen2[xn, 0], m.LLGen2[xn, 1]+1)) +
                sum(m.vFlow[self.connections['Flow'][xh] +
                            m.LLESec2[m.LLN2B1[x2+m.LLN2B2[xn, 1]], xs], xt] -
                    m.vLoss[self.connections['Loss'][xh] +
                            m.LLN2B1[x2+m.LLN2B2[xn, 1]], xt]/2
                    for x2 in range(1+m.LLN2B2[xn, 0])) ==
                self.busData[xn]*self.scenarios['Demand']
                                               [xt+self.busScenario[xn][xh]] -
                (m.vStore[self.LLStor[xn, xh], 0] -
                 m.vStore[self.LLStor[xn, xh], self.LLTime[xt]])*aux -
                m.vFea[self.connections['Feasibility'][xh]+m.LLFea[xn+1], xt] +
                m.vDL[self.connections['Pump'][xh]+m.LLDL[xn], xt] +
                sum(m.vFlow[self.connections['Flow'][xh] +
                            m.LLESec2[m.LLN2B1[x1+m.LLN2B2[xn, 3]], xs], xt] +
                    m.vLoss[self.connections['Loss'][xh] +
                            m.LLN2B1[x1+m.LLN2B2[xn, 3]], xt]/2
                    for x1 in range(1+m.LLN2B2[xn, 2])))

    # Power losses (Positive)
    def DCLossA_rule(self, m, xb, xb2, xt, xh):
        return (m.vLoss[self.connections['Loss'][xh]+xb+1, xt] >=
                (m.Loss_Con1[xb2]+m.vFlow[self.connections['Flow'][xh] +
                                          xb+1, xt]*m.Loss_Con2[xb2]) *
                m.branchData[xb, 0])

    # Power losses (Negative)
    def DCLossB_rule(self, m, xb, xb2, xt, xh):
        return (m.vLoss[self.connections['Loss'][xh]+xb+1, xt] >=
                (m.Loss_Con1[xb2] -
                 m.vFlow[self.connections['Flow'][xh]+xb+1, xt] *
                 m.Loss_Con2[xb2])*m.branchData[xb, 0])

    # No losses
    def DCLossN_rule(self, m, xb, xt, xh):
        return m.vLoss[self.connections['Loss'][xh]+xb+1, xt] == 0

    # Maximum capacity of dynamic loads
    def LDMax_rule(self, m, xdl, xt, xh):
        return m.vDL[self.connections['Pump'][xh]+xdl, xt] <= m.MaxDL[xdl]

    # Initialising dynamic loads
    def LDIni_rule(self, m, xt, xh):
        return m.vDL[self.connections['Pump'][xh], xt] == 0

    # Positions without feasibility constraints
    def setFea_rule(self, m, xt, xh):
        return m.vFea[self.connections['Feasibility'][xh], xt] == 0

    # Reference storage
    def Store0_rule(self, m, xt):
        return m.vStore[0, xt] == 0

    # Maximum storage
    def StoreMax_rule(self, m, xst, xt, xh):
        aux = xh*self.Storage['Number']+xst+1
        return m.vStore[aux, xt] <= self.Storage['Max'][xst]

    # Minimum storage
    def StoreMin_rule(self, m, xst, xt, xh):
        aux = xh*self.Storage['Number']+xst+1
        return m.vStore[aux, xt] >= 0.2*self.Storage['Max'][xst]

    #                                   Sets                                  #
    def getSets(self, m):
        m.sBra = range(self.networkE.number_of_edges())
        m.sBus = range(self.networkE.number_of_nodes())
        m.sTim = range(self.settings['NoTime'])
        m.sLoss = range(self.Number_LossCon)
        m.sBranch = range(self.NoBranch+1)
        m.sBuses = range(self.NoBuses+1)
        m.sN2B = range(self.NoN2B)
        m.sSec1 = range(self.NoSec1)
        m.sSec2 = range(self.NoSec2+1)
        m.sGen = range(self.generationE['Number'])
        m.sGenCM = range(self.NoGenC)
        m.sDL = range(self.pumps['Number'])
        m.sFea = range(self.NoFea)
        m.sSto = range(self.Storage['Number'])
        m.sCon = self.connections['set']

        return m

    #                                Parameters                               #
    def getPar(self, m):
        m.branchNo = self.branchNo
        m.branchData = self.branchData
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
        m.LLGenC = self.LLGenC
        m.GenLCst = self.GenLCst
        m.ValDL = self.pumps['Value']
        m.MaxDL = self.pumps['Max']
        m.LLDL = self.LLDL
        m.LLFea = self.LLFea

        return m

    #                             Model Variables                             #
    def getVars(self, m):
        Noh = len(m.sCon)
        m.vFlow = Var(range(Noh*(self.NoBranch+1)), m.sTim,
                             domain=Reals, initialize=0.0)
        m.vVolt = Var(range(Noh*(self.NoBuses+1)), m.sTim,
                               domain=Reals, initialize=0.0)
        m.vLoss = Var(range(Noh*(self.networkE.number_of_edges()+1)),
                             m.sTim, domain=NonNegativeReals, initialize=0.0)
        m.vFea = Var(range(Noh*self.NoFea), m.sTim, domain=NonNegativeReals,
                     initialize=0.0)
        m.vGen = Var(range(Noh*(self.generationE['Number']+1)), m.sTim, domain=NonNegativeReals,
                     initialize=0.0)
        m.vGCost = Var(range(Noh*self.generationE['Number']), m.sTim, domain=NonNegativeReals,
                       initialize=0.0)
        m.vDL = Var(range(Noh*(self.pumps['Number']+1)), m.sTim,
                       domain=NonNegativeReals, initialize=0.0)
        m.vStore = Var(range(Noh*(self.Storage['Number'])+1), m.sTim,
                       domain=NonNegativeReals, initialize=0.0)

        return m

    #                               Constraints                               #
    def addCon(self, m):
        # Reference line flow
        m.EPow0 = Constraint(m.sTim, m.sCon, rule=self.EPow0_rule)
        # Reference generation
        m.EGen0 = Constraint(m.sTim, m.sCon, rule=self.EGen0_rule)
        # Maximum generation
        m.EGMax = Constraint(m.sGen, m.sTim, m.sCon, rule=self.EGMax_rule)
        # Minimum generation
        m.EGMin = Constraint(m.sGen, m.sTim, m.sCon, rule=self.EGMin_rule)
        # Piece-wise generation costs approximation
        m.EGenC = Constraint(m.sGenCM, m.sTim, m.sCon, rule=self.EGenC_rule)
        # Branch flows
        m.EFlow = Constraint(m.sTim, m.sSec1, m.sCon, rule=self.EFlow_rule)
        # Branch capacity constraint (positive)
        m.EFMax = Constraint(m.sTim, m.sSec1, m.sCon, rule=self.EFMax_rule)
        # Branch capacity constraint (negative)
        m.EFMin = Constraint(m.sTim, m.sSec1, m.sCon, rule=self.EFMin_rule)
        # Balance: Generation + Flow in - loss/2 = Demand + flow out + loss/2
        m.EBalance = Constraint(m.sBus, m.sTim, m.sSec2, m.sCon,
                                rule=self.EBalance_rule)
        # Dinamic load maximum capacity
        m.DLMax = Constraint(m.sDL, m.sTim, m.sCon, rule=self.LDMax_rule)
        # Dinamic load initialisation
        m.DLIni = Constraint(m.sTim, m.sCon, rule=self.LDIni_rule)
        # Feasibility constraints
        m.setFea = Constraint(m.sTim, m.sCon, rule=self.setFea_rule)
        # Adding piece wise estimation of losses
        if self.settings['Losses']:
            m.DCLossA = Constraint(m.sBra, m.sLoss,
                                   m.sTim, m.sCon, rule=self.DCLossA_rule)
            m.DCLossB = Constraint(m.sBra, m.sLoss,
                                   m.sTim, m.sCon, rule=self.DCLossB_rule)
        else:
            m.DCLossNo = Constraint(m.sBra, m.sTim, m.sCon,
                                    rule=self.DCLossN_rule)
        # Adding RES limits
        if self.RES['Number'] > 0:
            m.RESMax = Constraint(m.sTim, range(self.RES['Number']),
                                  m.sCon,
                                  rule=self.RESMax_rule)

        # Storage
        if self.Storage['Number'] > 0:
            m.StoreMax = Constraint(m.sSto, m.sTim, m.sCon,
                                    rule=self.StoreMax_rule)
            m.StoreMin = Constraint(m.sSto, m.sTim, m.sCon,
                                    rule=self.StoreMin_rule)
        m.Store0 = Constraint(m.sTim, rule=self.Store0_rule)

        return m
