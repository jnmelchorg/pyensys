# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:04:58 2018

Pyene Networks provides methods to simulate power networks and generation units
(i.e., Unit commitment)

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
from __future__ import division
from pyomo.core import Constraint, Var, NonNegativeReals, Reals
import math
import numpy as np
import networkx as nx
import json
import warnings


class pyeneNConfig:
    ''' Default settings used for this class '''
    def __init__(self):
        # Basic settings
        self.settings = {
                'File': None,  # File to be loaded
                'Flag': True,  # Add electricity network
                'NoTime': 1,  # Number of time steps
                'Security': [],  # Security constraints (lines)
                'Losses': True,  # Consideration of losses
                'Feasibility': True,  # Feasibility constraints
                'Pieces': [],  # Size of pieces (MW) for piece-wise estimations
                'Constraint': [],  # Set line capacity constraints
                'GRamp': None,  # Set ramps for conventional generators
                'Loss': None,  # Factor for losses
                'Ancillary': None  # Need for uncillary services
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
                'Pump': [0]  # Pumps
                }
        # Scenarios
        self.scenarios = {
                'Number': 1,  # Number of scenarios
                'Links': 'Default',  # Links between the buses and profiles
                'NoDem': 0,  # Number of demand profiles
                'Demand': [1],  # Demand profiles
                'NoRES': 0,  # Number of RES profiles
                'LinksRes': 'Default',  # Links RES generators and profiles
                'RES': [],  # Location of the RES profiles
                'Weights': None  # Weight of the time period
                }
        # Conventional generators
        self.conventional = {
                'Number': None,  # Number of conventional generators
                'Ancillary': True,  # Can it provide ancillary services?
                'Ramp': None,  # Set ramps for conventional generators
                'RES': True  # Can it support RES integration?
                }
        # Hydropower
        self.hydropower = {
                'Number': 0,  # Number of hydropower plants
                'Bus': [],  # Location (Bus) in the network
                'Max': [],  # Capacity (MW)
                'Cost': [],  # Costs
                'Ramp': [],  # Ramp
                'Ancillary': True,  # Can it provide ancillary services?
                'RES': True,  # Can it support RES integration?
                'Link': None  # Position of hydropower plants
                }
        # Pumps
        self.pumps = {
                'Number': 0,  # Number of pumps
                'Bus': [],  # Location (Bus) in the network
                'Max': [],  # Capacity (kW)
                'Value': []  # Value (OF)
                }
        # RES
        self.RES = {
                'Number': 0,  # Number of RES generators
                'Bus': [],  # Location (Bus) in the network
                'Max': [],  # Capacity (kW)
                'Cost': [],  # Cost (OF)
                'Link': None,  # Position of RES generators
                'Uncertainty': None  # Introduce reserve needs
                }
        self.Storage = {
                'Number': 0,  # Number of storage units
                'Bus': [],  # Location (Bus) in the network
                'Max': [],  # Capacity (kW)
                'Efficiency': []  # Round trip efficiency
                }
        self.Print = {
                'Generation': True,
                'Flows': True,
                'Voltages': True,
                'Losses': True,
                'Curtailment': True,
                'Feasibility': True,
                }


class ENetworkClass:
    def __init__(self, obj=None):
        ''' Initialise network class '''
        # Get default values
        if obj is None:
            obj = pyeneNConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

        # sets and parameters used for the mathematical model
        self.s = {}
        self.p = {}

    def _getLL(self, NoLL, NoDt, DtLL):
        ''' Produce LL while removing the 'next' '''
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

    def addCon(self, m):
        ''' Add pyomo constraints '''

        if self.scenarios['NoDem'] == 0:
            self.p['daux'] = 0
        else:
            self.p['daux'] = 1
        if self.settings['Feasibility']:
            self.p['faux'] = 1
        else:
            self.p['faux'] = 0

        # Is the network enabled
        if self.settings['Flag']:
            # Reference line flow
            m.cNEPow0 = Constraint(self.s['Tim'], self.s['Con'],
                                   rule=self.cNEPow0_rule)
            # Branch flows
            m.cNEFlow = Constraint(self.s['Tim'], self.s['Sec1'],
                                   self.s['Con'], rule=self.cNEFlow_rule)
            # Branch capacity constraint (positive)
            m.cNEFMax = Constraint(self.s['Tim'], self.s['Sec1'],
                                   self.s['Con'], rule=self.cNEFMax_rule)
            # Branch capacity constraint (negative)
            m.cNEFMin = Constraint(self.s['Tim'], self.s['Sec1'],
                                   self.s['Con'], rule=self.cNEFMin_rule)
            # Adding piece wise estimation of losses
            if self.settings['Losses']:
                m.cNDCLossA = Constraint(self.s['Bra'], self.s['Loss'],
                                         self.s['Tim'], self.s['Con'],
                                         rule=self.cNDCLossA_rule)
                m.cNDCLossB = Constraint(self.s['Bra'], self.s['Loss'],
                                         self.s['Tim'], self.s['Con'],
                                         rule=self.cNDCLossB_rule)
            else:
                m.cNDCLossNo = Constraint(self.s['Bra'], self.s['Tim'],
                                          self.s['Con'],
                                          rule=self.cNDCLossN_rule)
            # Balance: Gen + Flow in - loss/2 = Demand + flow out + loss/2
            m.cNEBalance = Constraint(self.s['Bus'], self.s['Tim'],
                                      self.s['Sec2'], self.s['Con'],
                                      rule=self.cNEBalance_rule)
        else:
            # Create additional paremeters
            # Addition of power losses
            if self.settings['Loss'] is None:
                self.p['LossM'] = 1
            else:
                self.p['LossM'] = 1+self.settings['Loss']

            # Balance: Gen = Demand
            m.cNEBalance = Constraint(self.s['Tim'], self.s['Sec2'],
                                      self.s['Con'],
                                      rule=self.cNEBalance0_rule)

        # Reference generation
        m.cNEGen0 = Constraint(self.s['Tim'], self.s['Con'],
                               rule=self.cNEGen0_rule)
        # Maximum generation
        m.cNEGMax = Constraint(self.s['Gen'], self.s['Tim'], self.s['Con'],
                               rule=self.cNEGMax_rule)
        # Minimum generation
        m.cNEGMin = Constraint(self.s['Gen'], self.s['Tim'], self.s['Con'],
                               rule=self.cNEGMin_rule)
        # Piece-wise generation costs approximation
        m.cNEGenC = Constraint(self.s['GenCM'], self.s['Tim'], self.s['Con'],
                               rule=self.cNEGenC_rule)
        # Dinamic load maximum capacity
        m.cNDLMax = Constraint(self.s['Pump'], self.s['Tim'], self.s['Con'],
                               rule=self.cNLDMax_rule)
        # Dinamic load initialisation
        m.cNDLIni = Constraint(self.s['Tim'], self.s['Con'],
                               rule=self.cNLDIni_rule)
        # Feasibility constraints
        m.cNsetFea = Constraint(self.s['Tim'], self.s['Con'],
                                rule=self.cNsetFea_rule)
        # Adding RES limits
        if self.RES['Number'] > 0:
            m.cNRESMax = Constraint(self.s['Tim'], self.s['RES'],
                                    self.s['Con'], rule=self.cNRESMax_rule)
        # Storage
        if self.Storage['Number'] > 0:
            m.cNStoreMax = Constraint(m.sNSto, self.s['Tim'], self.s['Con'],
                                      rule=self.cNStoreMax_rule)
            m.cNStoreMin = Constraint(m.sNSto, self.s['Tim'], self.s['Con'],
                                      rule=self.cNStoreMin_rule)
        m.cNStore0 = Constraint(self.s['Tim'], rule=self.cNStore0_rule)

        # Adding generation ramps
        aux = len(self.s['GRamp'])
        if aux > 0:
            m.cNGenRampUp = \
                Constraint(range(aux), self.s['Tim'], self.s['Con'],
                           rule=self.cNGenRampUp_rule)
            m.cNGenRampDown = \
                Constraint(range(aux), self.s['Tim'], self.s['Con'],
                           rule=self.cNGenRampDown_rule)

        # Adding service constraints
        if len(self.s['GServices']) > 0:
            m.cNServices = Constraint(self.s['Tim'], self.s['Con'],
                                      rule=self.cNServices_rule)
            if len(self.s['GAncillary']):
                m.cNServicesA = Constraint(self.s['Tim'], self.s['Con'],
                                           rule=self.cNServicesA_rule)
                m.cNAncillary = Constraint(self.s['Tim'], self.s['Con'],
                                           rule=self.cNAncillary_rule)
            if len(self.s['GRES']):
                m.cNServicesR = Constraint(self.s['Tim'], self.s['Con'],
                                           rule=self.cNServicesR_rule)
                m.cNUncRES = Constraint(self.s['Tim'], self.s['Con'],
                                        rule=self.cNUncRES_rule)

        return m

    def addPar(self, m):
        ''' Add pyomo parameters '''
        self.p['MaxPump'] = self.pumps['Max']

        return m

    def addSets(self, m):
        ''' Add pyomo sets '''
        self.s['Con'] = self.connections['set']
        self.s['Bra'] = range(self.networkE.number_of_edges())
        self.s['Bus'] = range(self.networkE.number_of_nodes())
        self.s['Buses'] = range(self.NoBuses+1)
        self.s['Pump'] = range(self.pumps['Number'])
        self.s['Fea'] = range(self.NoFea)
        self.s['Gen'] = range(self.generationE['Number'])
        self.s['GenCM'] = range(self.NoGenC)
        self.s['Tim'] = range(self.settings['NoTime'])
        self.s['Loss'] = range(self.Number_LossCon)
        self.s['Sec1'] = range(self.NoSec1)
        self.s['Sec2'] = range(self.NoSec2+1)
        self.s['Sto'] = range(self.Storage['Number'])
        self.s['RES'] = range(self.RES['Number'])

        return m

    def addVars(self, m):
        ''' Add pyomo variables '''
        Noh = len(self.s['Con'])
        # Is the network enabled
        if self.settings['Flag']:
            m.vNFlow = Var(range(Noh*(self.NoBranch+1)), self.s['Tim'],
                           domain=Reals, initialize=0.0)
            m.vNLoss = Var(range(Noh*(self.networkE.number_of_edges()+1)),
                           self.s['Tim'], domain=NonNegativeReals,
                           initialize=0.0)
            m.vNVolt = Var(range(Noh*(self.NoBuses+1)), self.s['Tim'],
                           domain=Reals, initialize=0.0)

        m.vNPump = Var(range(Noh*(self.pumps['Number']+1)), self.s['Tim'],
                       domain=NonNegativeReals, initialize=0.0)
        m.vNFea = Var(range(Noh*self.NoFea), self.s['Tim'],
                      domain=NonNegativeReals, initialize=0.0)
        m.vNGCost = Var(range(Noh*self.generationE['Number']), self.s['Tim'],
                        domain=NonNegativeReals, initialize=0.0)
        m.vNGen = Var(range(Noh*(self.generationE['Number']+1)), self.s['Tim'],
                      domain=NonNegativeReals, initialize=0.0)
        m.vNStore = Var(range(Noh*(self.Storage['Number'])+1), self.s['Tim'],
                        domain=NonNegativeReals, initialize=0.0)
        if len(self.s['GServices']) > 0:
            m.vNServ = Var(range(Noh*self.p['GServices']), self.s['Tim'],
                           domain=NonNegativeReals, initialize=0.0)

        return m

    def cNServices_rule(self, m, xt, xh):
        ''' Provision of all services '''
        aux = xh*(self.generationE['Number']+1)+1
        return sum(m.vNGen[aux+x, xt] for x in self.s['GServices']) >= \
            sum(m.vNServ[xh*self.p['GServices']+xs, xt]
                for xs in range(self.p['GServices']))

    def cNServicesA_rule(self, m, xt, xh):
        ''' Provision of ancillary services '''
        aux = xh*(self.generationE['Number']+1)+1
        return sum(m.vNGen[aux+x, xt] for x in self.s['GAncillary']) >= \
            m.vNServ[xh*self.p['GServices'], xt]

    def cNServicesR_rule(self, m, xt, xh):
        ''' Provision of RES support services '''
        aux = xh*(self.generationE['Number']+1)+1
        return sum(m.vNGen[aux+x, xt] for x in self.s['GRES']) >= \
            m.vNServ[self.p['GServices']*(xh+1)-1, xt]

    def cNUncRES_rule(self, m, xt, xh):
        ''' Corrected maximum RES generation '''
        return sum(m.vNGen[self.resScenario[xg][xh][0], xt]
                   for xg in self.s['RES']) <= \
            sum(self.scenarios['RES'][self.resScenario[xg][xh][1]+xt] *
                self.RES['Max'][xg] for xg in self.s['RES']) * \
            (1-self.RES['Uncertainty'])+m.vNFea[xh*self.p['faux']+1, xt] + \
            m.vNServ[self.p['GServices']*(xh+1)-1, xt]

    def cNAncillary_rule(self, m, xt, xh):
        ''' Ancillary services constraint '''
        aux = xh*(self.generationE['Number']+1)+1
        return m.vNServ[xh*self.p['GServices'], xt] >= \
            self.settings['Ancillary'] * \
            sum(m.vNGen[aux+x, xt] for x in self.s['Gen']) - \
            m.vNFea[xh*self.p['faux'], xt]

    def cNDCLossA_rule(self, m, xb, xb2, xt, xh):
        ''' Power losses (Positive) '''
        return (m.vNLoss[self.connections['Loss'][xh]+xb+1, xt] >=
                (self.p['Loss_Con1'][xb2] +
                 m.vNFlow[self.connections['Flow'][xh] +
                          xb+1, xt]*self.p['Loss_Con2'][xb2]) *
                self.p['branchData'][xb, 0])

    def cNDCLossB_rule(self, m, xb, xb2, xt, xh):
        ''' Power losses (Negative) '''
        return (m.vNLoss[self.connections['Loss'][xh]+xb+1, xt] >=
                (self.p['Loss_Con1'][xb2] -
                 m.vNFlow[self.connections['Flow'][xh]+xb+1, xt] *
                 self.p['Loss_Con2'][xb2])*self.p['branchData'][xb, 0])

    def cNDCLossN_rule(self, m, xb, xt, xh):
        ''' No losses '''
        return m.vNLoss[self.connections['Loss'][xh]+xb+1, xt] == 0

    def cNEBalance_rule(self, m, xn, xt, xs, xh):
        ''' Nodal balance:
        Generation + Flow in - loss/2 = Demand + flow out + loss/2
        '''

        # Check for case without demand profiles
        if self.LLStor[xn, xh] == 0:
            aux = 0
        else:
            aux = self.Storage['Efficiency'][self.LLStor[xn, 0]-1]

        return (sum(m.vNGen[self.connections['Generation'][xh] +
                            self.p['LLGen1'][xg], xt]
                    for xg in range(self.p['LLGen2'][xn, 0],
                                    self.p['LLGen2'][xn, 1]+1)) +
                sum(m.vNFlow[self.connections['Flow'][xh] +
                             self.p['LLESec2'][self.p['LLN2B1']
                                               [x2+self.p['LLN2B2'][xn, 1]],
                                               xs], xt] -
                    m.vNLoss[self.connections['Loss'][xh] +
                             self.p['LLN2B1'][x2 +
                                              self.p['LLN2B2'][xn, 1]], xt]/2
                    for x2 in range(1+self.p['LLN2B2'][xn, 0])) ==
                self.busData[xn]*self.scenarios['Demand']
                                               [xt*self.p['daux'] +
                                                self.busScenario[xn][xh]] -
                (m.vNStore[self.LLStor[xn, xh], xt] -
                 m.vNStore[self.LLStor[xn, xh], self.LLTime[xt]])*aux /
                self.scenarios['Weights'][xt] -
                m.vNFea[self.connections['Feasibility'][xh] +
                        self.p['LLFea'][xn+1], xt] +
                m.vNPump[self.connections['Pump'][xh]+self.p['LLPump'][xn],
                         xt] +
                sum(m.vNFlow[self.connections['Flow'][xh] +
                             self.p['LLESec2'][self.p['LLN2B1']
                             [x1+self.p['LLN2B2'][xn, 3]], xs], xt] +
                    m.vNLoss[self.connections['Loss'][xh] +
                             self.p['LLN2B1'][x1 +
                                              self.p['LLN2B2'][xn, 3]], xt]/2
                    for x1 in range(1+self.p['LLN2B2'][xn, 2])))

    def cNEBalance0_rule(self, m, xt, xs, xh):
        ''' Nodal balance without networks '''
        # Check for case without demand profiles
        return sum(self.busData[xn]*self.scenarios['Demand']
                   [xt*self.p['daux']+self.busScenario[xn][xh]]
                   for xn in self.s['Bus'])*self.p['LossM'] + \
            + sum(m.vNPump[xh*(self.pumps['Number']+1)+xp+1, xt]
                  for xp in self.s['Pump']) == \
            m.vNFea[xh*self.p['faux'], xt] + \
            sum(m.vNGen[xh*(self.generationE['Number']+1)+xg+1, xt]
                for xg in self.s['Gen']) - \
            sum((m.vNStore[xh*(self.Storage['Number']+1)+xn+1, xt] -
                 m.vNStore[xh*(self.Storage['Number']+1)+xn+1,
                           self.LLTime[xt]])*self.Storage['Efficiency'][xn] /
                self.scenarios['Weights'][xt] for xn in self.s['Sto'])

    def cNEFlow_rule(self, m, xt, xb, xh):
        ''' Branch flows '''
        aux = self.connections['Voltage'][xh]+self.p['LLESec1'][xb, 1]
        xaux1 = aux+self.p['branchNo'][self.p['LLESec1'][xb, 0], 0]
        xaux2 = aux+self.p['branchNo'][self.p['LLESec1'][xb, 0], 1]
        return (m.vNFlow[self.connections['Flow'][xh]+xb+1, xt] ==
                (m.vNVolt[xaux1, xt]-m.vNVolt[xaux2, xt]) /
                self.p['branchData'][self.p['LLESec1'][xb, 0], 1])

    def cNEFMax_rule(self, m, xt, xb, xh):
        ''' Branch capacity constraint (positive) '''
        return (m.vNFlow[self.connections['Flow'][xh]+xb+1, xt] >=
                -self.p['branchData'][self.p['LLESec1'][xb, 0], 3])

    def cNEFMin_rule(self, m, xt, xb, xh):
        ''' Branch capacity constraint (negative) '''
        return (m.vNFlow[self.connections['Flow'][xh]+xb+1, xt] <=
                self.p['branchData'][self.p['LLESec1'][xb, 0], 3])

    def cNEGen0_rule(self, m, xt, xh):
        ''' Reference generation '''
        return m.vNGen[self.connections['Generation'][xh], xt] == 0

    def cNEGenC_rule(self, m, xc, xt, xh):
        ''' Piece-wise generation costs approximation '''
        return (m.vNGCost[self.connections['Cost'][xh] +
                          self.p['LLGenC'][xc], xt] /
                self.scenarios['Weights'][xt] >=
                m.vNGen[self.connections['Generation'][xh] +
                        self.p['LLGenC'][xc]+1, xt] *
                self.p['GenLCst'][xc, 0]+self.p['GenLCst'][xc, 1])

    def cNEGMax_rule(self, m, xg, xt, xh):
        ''' Maximum generation '''
        return (m.vNGen[self.connections['Generation'][xh]+xg+1, xt] <=
                self.p['GenMax'][xg])

    def cNEGMin_rule(self, m, xg, xt, xh):
        ''' Minimum generation '''
        return (m.vNGen[self.connections['Generation'][xh]+xg+1, xt] >=
                self.p['GenMin'][xg])

    def cNEPow0_rule(self, m, xt, xh):
        ''' Reference line flow '''
        return m.vNFlow[self.connections['Flow'][xh], xt] == 0

    def cNGenRampDown_rule(self, m, xg, xt, xh):
        ''' Generation ramps (down)'''
        x = xh*(self.generationE['Number']+1)+self.s['GRamp'][xg]+1
        return m.vNGen[x, xt]-m.vNGen[x, self.LLTime[xt]] >= \
            -self.p['GRamp'][xg]

    def cNGenRampUp_rule(self, m, xg, xt, xh):
        ''' Generation ramps (up)'''
        x = xh*(self.generationE['Number']+1)+self.s['GRamp'][xg]+1
        return m.vNGen[x, xt]-m.vNGen[x, self.LLTime[xt]] <= \
            self.p['GRamp'][xg]

    def cNLDIni_rule(self, m, xt, xh):
        ''' Initialising dynamic loads '''
        return m.vNPump[self.connections['Pump'][xh], xt] == 0

    def cNLDMax_rule(self, m, xdl, xt, xh):
        ''' Maximum capacity of dynamic loads'''
        return (m.vNPump[self.connections['Pump'][xh]+xdl+1, xt] <=
                self.p['MaxPump'][xdl]/self.networkE.graph['baseMVA'])

    def cNRESMax_rule(self, m, xt, xg, xh):
        ''' Maximum RES generation '''
        return (m.vNGen[self.resScenario[xg][xh][0], xt] <=
                self.scenarios['RES'][self.resScenario[xg][xh][1]+xt] *
                self.RES['Max'][xg])

    def cNsetFea_rule(self, m, xt, xh):
        ''' Positions without feasibility constraints '''
        return m.vNFea[self.connections['Feasibility'][xh], xt] == 0

    def cNStore0_rule(self, m, xt):
        ''' Reference storage '''
        return m.vNStore[0, xt] == 0

    def cNStoreMax_rule(self, m, xst, xt, xh):
        ''' Maximum storage '''
        aux = xh*self.Storage['Number']+xst+1
        return m.vNStore[aux, xt] <= self.Storage['Max'][xst]

    def cNStoreMin_rule(self, m, xst, xt, xh):
        ''' Minimum storage '''
        aux = xh*self.Storage['Number']+xst+1
        return m.vNStore[aux, xt] >= 0.2*self.Storage['Max'][xst]

    def initialise(self):
        ''' Initialize externally '''
        # Setting additional constraints (Security, losses and feasibilty)

        # Read network data
        self.Read()

        self.ProcessENet()

        (self.busData, self.busScenario,
         self.resScenario) = self.ProcessEDem(self.demandE)

        self.ProcessEGen()

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

        # Initialise weights per scenario
        if self.scenarios['Weights'] is None:
            self.scenarios['Weights'] = np.ones(self.settings['NoTime'],
                                                dtype=float)
        # Sets and parameters for modelling ramp constraints
        if self.conventional['Ramp'] is not None:
            if len(self.hydropower['Ramp']) > 0:
                # Conventional and hydropower
                aux = self.conventional['Number']+self.hydropower['Number']
                xh = self.conventional['Number']
            else:
                # Only conventional
                aux = self.conventional['Number']

            self.s['GRamp'] = np.zeros(aux, dtype=int)
            self.p['GRamp'] = np.zeros(aux, dtype=float)
            for xg in range(self.conventional['Number']):
                self.s['GRamp'][xg] = xg
                self.p['GRamp'][xg] = \
                    self.conventional['Ramp']*self.p['GenMax'][xg]
        else:
            if len(self.hydropower['Ramp']) > 0:
                # Only hydropower
                aux = self.hydropower['Number']
                xh = 0
                self.s['GRamp'] = np.zeros(aux, dtype=int)
                self.p['GRamp'] = np.zeros(aux, dtype=float)
            else:
                self.s['GRamp'] = []
                self.p['GRamp'] = []

        if len(self.hydropower['Ramp']) > 0:
            for xg in range(self.hydropower['Number']):
                self.s['GRamp'][xh] = self.conventional['Number']+xg
                self.p['GRamp'][xh] = self.hydropower['Ramp'][xg] * \
                    self.hydropower['Max'][xg]/self.networkE.graph['baseMVA']
                xh += 1

        # Sets and parameters for modelling Ancilarry service requirements
        NoSer = 0
        self.s['GAncillary'] = []
        if self.settings['Ancillary'] is not None:
            # Check for units that can provide ancillary services
            if self.conventional['Ancillary']:
                NoSer += 1
                if self.hydropower['Ancillary']:
                    ''' Conv and hydro can provide ancillary services '''
                    aux = self.conventional['Number']+self.hydropower['Number']
                    xh = self.conventional['Number']
                else:
                    # Only conventional
                    aux = self.conventional['Number']

                self.s['GAncillary'] = np.zeros(aux, dtype=int)
                for xg in range(self.conventional['Number']):
                    self.s['GAncillary'][xg] = xg
            else:
                if self.hydropower['Ancillary']:
                    # Only hydro can provide ancillary services
                    NoSer += 1
                    self.s['GAncillary'] = np.zeros(self.hydropower['Number'],
                                                    dtype=int)
                    xh = 0
                else:
                    # There are no means to provide ancillary services
                    warnings.warn('Warning: Unable to provide'
                                  ' ancillary services')

        if self.settings['Ancillary'] is not None and \
                self.hydropower['Ancillary']:
            for xg in range(self.hydropower['Number']):
                self.s['GAncillary'][xh] = self.conventional['Number']+xg
                xh += 1

        # Sets and parameters for modelling RES support
        self.s['GRES'] = []
        if self.RES['Number'] > 0 and self.RES['Uncertainty'] is not None:
            # Check for units that can provide RES support
            if self.conventional['RES']:
                NoSer += 1
                if self.hydropower['RES']:
                    ''' Conv and hydro can provide ancillary services '''
                    aux = self.conventional['Number']+self.hydropower['Number']
                    xh = self.conventional['Number']
                else:
                    # Only conventional
                    aux = self.conventional['Number']

                self.s['GRES'] = np.zeros(aux, dtype=int)
                for xg in range(self.conventional['Number']):
                    self.s['GRES'][xg] = xg
            else:
                if self.hydropower['RES']:
                    NoSer += 1
                    # Only hydro can provide RES services
                    self.s['GRES'] = np.zeros(self.hydropower['Number'],
                                              dtype=int)
                    xh = 0

            if self.hydropower['RES']:
                for xg in range(self.hydropower['Number']):
                    self.s['GRES'][xh] = self.conventional['Number']+xg
                    xh += 1

        # Generators providing services
        self.s['GServices'] = np.unique(np.concatenate((self.s['GAncillary'],
                                                        self.s['GRES']), 0))
        self.p['GServices'] = NoSer

    def OF_rule(self, m):
        ''' Objective function '''
        xh = self.connections['set'][0]
        return (sum(sum(m.vNGCost[self.connections['Cost'][xh]+xg, xt]
                        for xg in self.s['Gen']) +
                    sum(m.vNFea[self.connections['Feasibility'][xh]+xf, xt]
                        for xf in self.s['Fea']) *
                    1000000 for xt in self.s['Tim']) -
                sum(self.pumps['Value'][xdl]*self.networkE.graph['baseMVA'] *
                    sum(m.vNPump[self.connections['Pump'][xh] +
                                 xdl+1, xt]*self.scenarios['Weights'][xt]
                        for xt in self.s['Tim']) for xdl in self.s['Pump']))

    def offPrint(self):
        ''' Switch off print flags '''
        self.Print['Generation'] = False
        self.Print['Flows'] = False
        self.Print['Voltages'] = False
        self.Print['Losses'] = False
        self.Print['Curtailment'] = False
        self.Print['Feasibility'] = False

    def print(self, m, sh=None):
        ''' Print results '''
        if sh is None:
            sh = self.s['Con']

        for xh in sh:
            print("\nCASE:", xh)

            if self.Print['Generation']:
                print("\nFlow_EGen=[")
                for xn in range(1, self.generationE['Number']+1):
                    for x2 in self.s['Tim']:
                        aux = (m.vNGen[self.connections['Generation'][xh]+xn,
                                       x2].value *
                               self.networkE.graph['baseMVA'])
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Flows'] and self.settings['Flag']:
                print("\nFlow_EPower=[")
                for x1 in range(1, self.networkE.number_of_edges()+1):
                    for x2 in self.s['Tim']:
                        aux = (m.vNFlow[self.connections['Flow'][xh] +
                                        x1, x2].value *
                               self.networkE.graph['baseMVA'])
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Voltages'] and self.settings['Flag']:
                print("\nVoltage_Angle=[")
                for xn in self.s['Buses']:
                    for xt in self.s['Tim']:
                        aux = self.connections['Voltage'][xh]
                        aux = m.vNVolt[aux+xn, xt].value
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Losses'] and self.settings['Flag']:
                print("\nEPower_Loss=[")
                for xb in range(1, self.networkE.number_of_edges()+1):
                    for xt in self.s['Tim']:
                        aux = (m.vNLoss[self.connections['Loss'][xh]+xb,
                                        xt].value *
                               self.networkE.graph['baseMVA'])
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Curtailment']:
                print("\nPumps=[")
                for xdl in self.s['Pump']:
                    for xt in self.s['Tim']:
                        aux = m.vNPump[self.connections['Pump'][xh]+xdl+1,
                                       xt].value*self.networkE.graph['baseMVA']
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Feasibility']:
                print("\nFeas=[")
                for xf in self.s['Fea']:
                    for xt in self.s['Tim']:
                        aux = m.vNFea[self.connections['Feasibility'][xh]+xf,
                                      xt].value*self.networkE.graph['baseMVA']
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

    def ProcessEDem(self, ENetDem):
        ''' Process demand and generation parameters '''
        # Adjust demand profiles
        busData = np.zeros(self.networkE.number_of_nodes(), dtype=float)
        for xn in range(self.networkE.number_of_nodes()):
            busData[xn] = self.demandE['PD'][xn]/self.networkE.graph['baseMVA']
        # Auxiliar to find demand profiles
        busScenario = np.zeros((self.networkE.number_of_nodes(),
                                self.scenarios['Number']), dtype=int)

        if self.scenarios['NoDem'] > 0:
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

    def ProcessEGen(self):
        ''' Process generator parameters '''
        GenMax = self.generationE['Data']['PMAX']
        GenMin = self.generationE['Data']['PMIN']

        # Get LL for generators
        (LLGen1, LLGen2) = self._getLL(self.networkE.number_of_nodes(),
                                       self.generationE['Number'],
                                       self.generationE['Data']['GEN_BUS'])
        self.p['LLGen1'] = LLGen1
        self.p['LLGen2'] = LLGen2

        # Get size of the required pieces
        laux = len(self.settings['Pieces'])
        vaux = self.settings['Pieces']
        self.settings['Pieces'] = np.zeros(self.generationE['Number'],
                                           dtype=int)
        # Set selected value for conventional generators
        if laux == 1:
            for xg in range(self.conventional['Number']):
                self.settings['Pieces'][xg] = vaux[0]
        # Set selected values for the first set of generators
        elif laux < self.generationE['Number']:
            for xg in range(laux):
                self.settings['Pieces'][xg] = vaux[xg]

        # Get number of variables and differentials required
        pwNo = np.zeros(self.generationE['Number'], dtype=int)
        NoGenC = 0
        for xg in range(self.generationE['Number']):
            # Linear model - single piece
            if self.generationE['Costs']['MODEL'][xg] == 1:
                NoGenC += self.generationE['Costs']['NCOST'][xg]
            # Quadratic with default settings
            elif self.settings['Pieces'][xg] == 0:
                pwNo[xg] = self.generationE['Costs']['NCOST'][xg]
                NoGenC += pwNo[xg]
            # Quadratic with bespoke number of pieces
            elif self.generationE['Costs']['MODEL'][xg] == 2:
                pwNo[xg] = math.ceil((self.generationE['Data']
                                      ['PMAX'][xg] -
                                      self.generationE['Data']
                                      ['PMIN'][xg]) /
                                     self.settings['Pieces'][xg])
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
        self.p['LLGenC'] = LLGenC

        # Changing to pu
        for xg in range(self.generationE['Number']):
            GenMax[xg] /= self.networkE.graph['baseMVA']
            GenMin[xg] /= self.networkE.graph['baseMVA']
        for xc in range(NoGenC):
            GenLCst[xc][0] *= self.networkE.graph['baseMVA']
        self.p['GenMax'] = GenMax
        self.p['GenMin'] = GenMin
        self.p['GenLCst'] = GenLCst
        self.NoGenC = NoGenC

    def ProcessENet(self):
        ''' Process information for optimisation purposes '''
        # Map connections between nodes and branches (non-sequential search)
        NoN2B = self.networkE.number_of_edges()*2+1  # Number of data points
        LLaux = np.zeros(NoN2B, dtype=int)  # connections (non-sequential)
        LLnext = np.zeros(NoN2B, dtype=int)  # Next connection (non-sequential)
        LLN2B1 = np.zeros(NoN2B, dtype=int)  # connections (sequential)
        # Position of first connection and number of connections
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
        self.NoN2B = NoN2B
        self.p['LLN2B1'] = LLN2B1
        self.p['LLN2B2'] = LLN2B2

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

        self.p['branchNo'] = branchNo
        self.p['branchData'] = branchData

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
        self.NoSec1 = NoSec1
        self.NoSec2 = NoSec2
        self.p['LLESec1'] = LLESec1
        self.p['LLESec2'] = LLESec2

        # Adjust branch data if there are predefined constraints
        aux = len(self.settings['Constraint'])
        if aux > 0:
            if aux == 1:
                aux = self.settings['Constraint'] / \
                    self.networkE.graph['baseMVA']
                self.settings['Constraint'] = np.zeros(branchNo, dtype=float)
                for xb in range(branchNo):
                    self.settings['Constraint'][xb] = aux
            for xb in range(branchNo):
                branchData[LLESec1[xb][0]][3] = self.settings['Constraint'][xb]

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
        self.Number_LossCon = Number_LossCon
        self.p['Loss_Con1'] = Loss_Con1
        self.p['Loss_Con2'] = Loss_Con2

        # Add LL for dynamic loads
        LLDL = np.zeros(self.networkE.number_of_nodes(), dtype=int)
        for xdl in range(self.pumps['Number']):
            LLDL[self.pumps['Bus'][xdl]-1] = xdl+1
        self.p['LLPump'] = LLDL

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
        self.p['LLFea'] = LLFea
        self.NoFea = NoFea

    def Read(self):
        ''' Read input data '''
        # Load file
        mpc = json.load(open(self.settings['File']))

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
                'NCOST': np.zeros(NoGen, dtype=int),
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
        # All devices are linked to the same profile
        if self.scenarios['NoRES'] == 1:
            self.scenarios['LinksRes'] = np.ones(self.scenarios['Number'] *
                                                 self.RES['Number'], dtype=int)
        # i.e., each scenario is linked to a profile
        elif self.scenarios['LinksRes'] == 'Default':
            self.scenarios['LinksRes'] = np.ones(self.scenarios['Number'] *
                                                 self.RES['Number'], dtype=int)
            acu = self.RES['Number']
            for xs in range(self.scenarios['Number']-1):
                for xt in range(self.RES['Number']):
                    self.scenarios['LinksRes'][acu] = xs+2
                    acu += 1

        # Add renewable generation
        self.conventional['Number'] = NoOGen
        if self.hydropower['Number'] > 0:
            # Adding hydro
            self.hydropower['Link'] = range(NoOGen,
                                            NoOGen+self.hydropower['Number'])
            ENetGen['GEN_BUS'][self.hydropower['Link']] = (self.hydropower
                                                           ['Bus'])
            ENetGen['PMAX'][self.hydropower['Link']] = self.hydropower['Max']
            xg2 = -1
            for xg in self.hydropower['Link']:
                xg2 += 1
                ENetGen['MBASE'][xg] = self.networkE.graph['baseMVA']
                # Add polinomial (linear) cost curve
                if self.hydropower['Cost'][xg2] != 0:
                    ENetCost['MODEL'][xg] = 2
                    ENetCost['NCOST'][xg] = 2
                    ENetCost['COST'][xg][0] = self.hydropower['Cost'][xg2]

        if self.RES['Number'] > 0:
            # Adding intermittent generation
            self.RES['Link'] = range(NoOGen+self.hydropower['Number'], NoGen)
            ENetGen['GEN_BUS'][self.RES['Link']] = self.RES['Bus']
            xg2 = -1
            for xg in self.RES['Link']:
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

#        # Creating conventional generator objects
#        from .pyeneD import ConvClass
#        DM_GenC = [ConvClass() for x in range(NoOGen)]
#        scens = np.zeros(self.scenarios['Number'], dtype=int)
#        for xs in range(1, self.scenarios['Number']):
#            scens[xs] = xs*NoGen
#        acu = 1
#        for xc in range(NoOGen):
#            # Connection to the electricity network
#            DM_GenC[xc].name = 'Conventional generator No. ' + str(xc)
#            DM_GenC[xc].node['NM'] = mpc['gen']['GEN_BUS'][xc]
#            DM_GenC[xc].gen['PG'] = mpc['gen']['PG'][xc]
#            DM_GenC[xc].gen['QG'] = mpc['gen']['QG'][xc]
#            DM_GenC[xc].gen['QMAX'] = mpc['gen']['QMAX'][xc]
#            DM_GenC[xc].gen['QMIN'] = mpc['gen']['QMIN'][xc]
#            DM_GenC[xc].gen['VG'] = mpc['gen']['VG'][xc]
#            DM_GenC[xc].gen['MBASE'] = mpc['gen']['MBASE'][xc]
#            DM_GenC[xc].gen['GEN'] = mpc['gen']['GEN'][xc]
#            DM_GenC[xc].gen['PMAX'] = mpc['gen']['PMAX'][xc]
#            DM_GenC[xc].gen['PMIN'] = mpc['gen']['PMIN'][xc]
#            DM_GenC[xc].gen['PC1'] = mpc['gen']['PC1'][xc]
#            DM_GenC[xc].gen['PC2'] = mpc['gen']['PC2'][xc]
#            DM_GenC[xc].gen['QC1MIN'] = mpc['gen']['QC1MIN'][xc]
#            DM_GenC[xc].gen['QC1MAX'] = mpc['gen']['QC1MAX'][xc]
#            DM_GenC[xc].gen['QC2MIN'] = mpc['gen']['QC2MIN'][xc]
#            DM_GenC[xc].gen['QC2MAX'] = mpc['gen']['QC2MAX'][xc]
#            DM_GenC[xc].gen['RAMP_AGC'] = mpc['gen']['RAMP_AGC'][xc]
#            DM_GenC[xc].gen['RAMP_10'] = mpc['gen']['RAMP_10'][xc]
#            DM_GenC[xc].gen['RAMP_30'] = mpc['gen']['RAMP_30'][xc]
#            DM_GenC[xc].gen['RAMP_Q'] = mpc['gen']['RAMP_Q'][xc]
#            DM_GenC[xc].gen['APF'] = mpc['gen']['APF'][xc]
#            DM_GenC[xc].gen['MODEL'] = mpc['gencost']['MODEL'][xc]
#            DM_GenC[xc].gen['STARTUP'] = mpc['gencost']['STARTUP'][xc]
#            DM_GenC[xc].gen['SHUTDOWN'] = mpc['gencost']['SHUTDOWN'][xc]
#            DM_GenC[xc].gen['NCOST'] = mpc['gencost']['NCOST'][xc]
#            DM_GenC[xc].gen['COST'] = mpc['gencost']['COST'][xc][:]
#            DM_GenC[xc].no['vNGen'] = acu
#            DM_GenC[xc].no['vNGCost'] = acu
#            acu += 1
#
#        # Creating RES generator objects
#        from .pyeneD import RESClass
#        DM_GenR = [RESClass() for x in range(self.RES['Number'])]
#        for xr in range(self.RES['Number']):
#            # Connection to the electricity network
#            DM_GenR[xr].name = 'RES generator No. ' + str(xr)
#            DM_GenR[xr].node['NM'] = self.RES['Bus'][xr]
#            DM_GenR[xr].gen['PMAX'] = 1000000
#            DM_GenR[xr].gen['QMAX'] = 1000000
#            DM_GenR[xr].gen['QMIN'] = 0
#            DM_GenR[xr].gen['MODEL'] = 2
#            DM_GenR[xr].gen['NCOST'] = 2
#            DM_GenR[xr].gen['COST'] = self.RES['Cost'][xr]
#            DM_GenR[xr].gen['MBASE'] = self.networkE.graph['baseMVA']
#            DM_GenR[xr].no['vNGen'] = acu
#            DM_GenR[xr].no['vNGCost'] = acu
#            acu += 1
#
#        # Creating hydropower generators
#        from .pyeneD import HydroClass
#        DM_GenH = [HydroClass() for x in range(self.hydropower['Number'])]
#        for xh in range(self.hydropower['Number']):
#            DM_GenH[xh].name = 'Hydropower generator No. ' + str(xh)
#            DM_GenH[xh].node['NM'] = self.hydropower['Bus'][xh]
#            DM_GenH[xh].gen['PMAX'] = self.hydropower['Max'][xh]
#            DM_GenH[xh].gen['MBASE'] = self.networkE.graph['baseMVA']
#            DM_GenH[xh].gen['MODEL'] = 2
#            DM_GenH[xh].gen['NCOST'] = 2
#            DM_GenH[xh].gen['COST'] = self.hydropower['Cost'][xh]
#            DM_GenR[xc].no['vNGen'] = acu
#            DM_GenR[xc].no['vNGCost'] = acu
#            acu += 1
