# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:04:58 2018

Pyene Networks provides methods to simulate power networks and generation units
(i.e., Unit commitment)

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
from __future__ import division
from pyomo.core import Constraint, Var, NonNegativeReals, Reals, Binary
import math
import numpy as np
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
                'SecurityFlag': False,  # Enable all security constraints
                'Security': [],  # Security constraints (lines)
                'Losses': True,  # Consideration of losses
                'Feasibility': True,  # Feasibility constraints
                'Pieces': [],  # Size of pieces (MW) for piece-wise estimations
                'Loss': None,  # Factor for losses
                'Ancillary': None,  # Need for uncillary services
                'UC': False  # Model UC
                }
        # Connections
        self.connections = {
                'set': range(1),  # Connections between nodes
                'Branches': 0,  # Real number of branches (including parallel)
                'Flow': [0],  # Power flow through the lines
                'Voltage': [0],  # Voltages in each node
                'Loss': [0],  # Power losses
                'Loss_Param': [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2,
                               1.6, 2, 2.5, 3],  # To model power losses
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
                'Ancillary': [True],  # Can it provide ancillary services?
                'Baseload': [0],  # 0-1 for the use of conv for baseload
                'Ramp': [None],  # Set ramps for conventional generators
                'RES': [True]  # Can it support RES integration?
                }
        # Hydropower
        self.hydropower = {
                'Number': 0,  # Number of hydropower plants
                'Bus': [],  # Location (Bus) in the network
                'Max': [],  # Capacity (MW)
                'Cost': [],  # Costs
                'Ramp': [None],  # Ramp
                'Baseload': [0],  # 0-1 for the use of water for baseload
                'Ancillary': [True],  # Can it provide ancillary services?
                'RES': [True],  # Can it support RES integration?
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
                'Uncertainty': [None]  # Introduce reserve needs
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
                'Services': True,
                'GenBus': True,
                'UC': True,
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
            # Branch flows
            m.cNEFlow = Constraint(self.s['Tim'], self.s['Bra'],
                                   self.s['Sec2'], self.s['Con'],
                                   rule=self.cNEFlow_rule)
            # Branch capacity (Positive)
            m.cNEFMax = Constraint(self.s['Tim'], self.s['Bra'],
                                   self.s['Sec2'], self.s['Con'],
                                   rule=self.cNEFMax_rule)
            # Branch capacity (Negative)
            m.cNEFMin = Constraint(self.s['Tim'], self.s['Bra'],
                                   self.s['Sec2'], self.s['Con'],
                                   rule=self.cNEFMin_rule)

            if self.settings['Losses']:
                m.cNDCLossA = Constraint(self.s['Bra'], self.s['Loss'],
                                         self.s['Tim'], self.s['Con'],
                                         rule=self.cNDCLossA_rule)
                m.cNDCLossB = Constraint(self.s['Bra'], self.s['Loss'],
                                         self.s['Tim'], self.s['Con'],
                                         rule=self.cNDCLossB_rule)

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
            m.cNEBalance0 = Constraint(self.s['Tim'], self.s['Sec2'],
                                       self.s['Con'],
                                       rule=self.cNEBalance0_rule)
        # Is there baseload
        if sum(self.hydropower['Baseload']) > 0:
            self.scenarios['WSum'] = sum(self.scenarios['Weights'])
            m.cNEBaseload = Constraint(self.s['Gen'], self.s['Tim'],
                                       self.s['Con'],
                                       rule=self.cNEBaseload_rule)
        if self.settings['UC']:
            # Maximum generation
            m.cNEGMax = Constraint(self.s['Gen'], self.s['Tim'], self.s['Con'],
                                   rule=self.cNEGMaxUC_rule)
            # Minimum generation
            m.cNEGMin = Constraint(self.s['Gen'], self.s['Tim'], self.s['Con'],
                                   rule=self.cNEGMinUC_rule)
        else:
            # Maximum generation
            m.cNEGMax = Constraint(self.s['Gen'], self.s['Tim'], self.s['Con'],
                                   rule=self.cNEGMax_rule)
            # Minimum generation
            m.cNEGMin = Constraint(self.s['Gen'], self.s['Tim'], self.s['Con'],
                                   rule=self.cNEGMin_rule)

        # Piece-wise generation costs approximation
        m.cNEGenC = Constraint(self.s['Gen'], range(self.Gen.get_NoPieces()),
                               self.s['Tim'], self.s['Con'],
                               rule=self.cNEGenC_rule)

        # Dinamic load (pump) maximum capacity
        m.cNDLMax = Constraint(self.s['Pump'], self.s['Tim'], self.s['Con'],
                               rule=self.cNLDMax_rule)
        # Dinamic load (pump) initialisation
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
            if len(self.s['GAncillary']) > 0:
                m.cNServicesA = Constraint(self.s['Tim'], self.s['Con'],
                                           rule=self.cNServicesA_rule)
                m.cNAncillary = Constraint(self.s['Tim'], self.s['Con'],
                                           rule=self.cNAncillary_rule)

            if len(self.s['GRES']) > 0:
                m.cNServicesR = Constraint(self.s['Tim'], self.s['Con'],
                                           rule=self.cNServicesR_rule)
                m.cNUncRES = Constraint(self.s['Tim'], self.s['Con'],
                                        rule=self.cNUncRES_rule)

        if len(self.s['GRES']) == 0 and self.RES['Number'] > 0 and \
                self.RES['Uncertainty'][0] is not None:
            m.cNUncRES0 = Constraint(self.s['Tim'], self.s['Con'],
                                     rule=self.cNUncRES0_rule)

        return m

    def addPar(self, m):
        ''' Add pyomo parameters '''
        self.p['MaxPump'] = self.pumps['Max']

        return m

    def addSets(self, m):
        ''' Add pyomo sets '''
        self.s['Con'] = self.connections['set']
        self.s['Bra'] = range(self.connections['Branches'])
        self.s['Bus'] = range(self.ENetwork.get_NoBus())
        self.s['Buses'] = range(self.NoBuses)
        self.s['Pump'] = range(self.pumps['Number'])
        self.s['Fea'] = range(self.NoFea)
        self.s['Gen'] = range(self.generationE['Number'])
        self.s['GenCM'] = range(self.NoGenC)
        self.s['Tim'] = range(self.settings['NoTime'])
        self.s['Sec1'] = range(self.NoSec1)
        self.s['Sec2'] = range(self.NoSec2+1)
        self.s['Sto'] = range(self.Storage['Number'])
        self.s['RES'] = range(self.RES['Number'])

        if self.settings['Losses']:
            self.s['Loss'] = range(self.Number_LossCon)

        return m

    def addVars(self, m):
        ''' Add pyomo variables '''
        Noh = len(self.s['Con'])
        # Is the network enabled
        if self.settings['Flag']:
            m.vNFlow = Var(range(Noh*(self.NoBranch)), self.s['Tim'],
                           domain=Reals, initialize=0.0)
            m.vNVolt = Var(range(Noh*(self.NoBuses)), self.s['Tim'],
                           domain=Reals, initialize=0.0)
            if self.settings['Losses']:
                m.vNLoss = Var(range(Noh*(self.connections['Branches'])),
                               self.s['Tim'], domain=NonNegativeReals,
                               initialize=0.0)

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
        # Add Unit commitment constraints
        if self.settings['UC']:
            m.vNGen_Bin = Var(range(Noh*(self.Gen.get_NoBin()+1)),
                              self.s['Tim'], domain=Binary, initialize=0.0)

        return m

    def cNAncillary_rule(self, m, xt, xh):
        ''' Ancillary services constraint '''
        return m.vNServ[xh*self.p['GServices'], xt] >= \
            self.settings['Ancillary'] * \
            sum(m.vNGen[self.connections['Generation'][xh]+x, xt]
                for x in self.Gen.get_GenAll())-m.vNFea[xh*self.p['faux'], xt]

    def cNDCLossA_rule(self, m, xb, xL, xt, xh):
        ''' Power losses (Positive) '''
        return self.ENetwork.cNDCLossA_rule(m, xt, xb, xL,
                                            self.connections['Flow'][xh],
                                            self.connections['Loss'][xh])

    def cNDCLossB_rule(self, m, xb, xL, xt, xh):
        ''' Power losses (Negative) '''
        return self.ENetwork.cNDCLossB_rule(m, xt, xb, xL,
                                            self.connections['Flow'][xh],
                                            self.connections['Loss'][xh])

    def cNEBalance_rule(self, m, xn, xt, xs, xh):
        ''' Nodal balance:
        Generation + Flow in - loss/2 = Demand + flow out + loss/2
        '''
        # Check for case without demand profiles
        if self.LLStor[xn, xh] == 0:
            aux = 0
        else:
            aux = self.Storage['Efficiency'][self.LLStor[xn, 0]-1]

        return (sum(m.vNGen[self.connections['Generation'][xh]+xg, xt]
                    for xg in self.Gen.get_GenInBus(self.ENetwork.Bus[xn])) +
                sum(m.vNFlow[self.connections['Flow'][xh]+x2, xt]
                    for x2 in self.ENetwork.get_FlowT(xn, xs)) -
                sum(m.vNLoss[self.connections['Loss'][xh]+x2, xt]/2
                    for x2 in self.ENetwork.Bus[xn].get_LossT()) ==
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
                sum(m.vNFlow[self.connections['Flow'][xh]+x1, xt]
                    for x1 in self.ENetwork.get_FlowF(xn, xs)) +
                sum(m.vNLoss[self.connections['Loss'][xh]+x1, xt]/2
                    for x1 in self.ENetwork.Bus[xn].get_LossF()))

    def cNEBalance0_rule(self, m, xt, xs, xh):
        ''' Nodal balance without networks '''
        return float(sum(self.busData[xn]*self.scenarios['Demand']
                     [xt*self.p['daux']+self.busScenario[xn][xh]]
                     for xn in self.s['Bus']))*self.p['LossM'] + \
            + sum(m.vNPump[xh*(self.pumps['Number']+1)+xp+1, xt]
                  for xp in self.s['Pump']) == \
            m.vNFea[xh*self.p['faux'], xt] + \
            sum(m.vNGen[self.connections['Generation'][xh]+xg, xt]
                for xg in self.Gen.get_GenAll()) - \
            sum((m.vNStore[xh*(self.Storage['Number']+1)+xn+1, xt] -
                 m.vNStore[xh*(self.Storage['Number']+1)+xn+1,
                           self.LLTime[xt]])*self.Storage['Efficiency'][xn] /
                self.scenarios['Weights'][xt] for xn in self.s['Sto'])

    def cNEBaseload_rule(self, m, xg, xt, xh):
        ''' Baseload rule '''
        return self.Gen.cNEBaseload_rule(m, xg, xt, self.s['Tim'],
                                         self.scenarios['Weights'],
                                         self.scenarios['WSum'],
                                         self.connections['Generation'][xh])
    
    def cNEFlow_rule(self, m, xt, xb, xs, xh):
        ''' Branch flows - DC model '''
        return self.ENetwork.cNEFlow_rule(m, xt, xb, xs,
                                          self.connections['Flow'][xh],
                                          self.connections['Voltage'][xh])

    def cNEFMax_rule(self, m, xt, xb, xs, xh):
        ''' Branch capacity constraint (positive) '''
        return self.ENetwork.cNEFMax_rule(m, xt, xb, xs,
                                          self.connections['Flow'][xh])

    def cNEFMin_rule(self, m, xt, xb, xs, xh):
        ''' Branch capacity constraint (positive) '''
        return self.ENetwork.cNEFMin_rule(m, xt, xb, xs,
                                          self.connections['Flow'][xh])

    def cNEGenC_rule(self, m, xg, xc, xt, xh):
        return self.Gen.cNEGenC_rule(m, xg, xc, xt,
                                     self.connections['Cost'][xh],
                                     self.connections['Generation'][xh],
                                     self.scenarios['Weights'][xt])

    def cNEGMax_rule(self, m, xg, xt, xh):
        ''' Maximum generation capacity '''
        return self.Gen.cNEGMax_rule(m, xg, xt,
                                     self.connections['Generation'][xh])

    def cNEGMin_rule(self, m, xg, xt, xh):
        ''' Minimum generation capacity '''
        return self.Gen.cNEGMin_rule(m, xg, xt,
                                     self.connections['Generation'][xh])

    def cNEGMaxUC_rule(self, m, xg, xt, xh):
        ''' Maximum generation '''
        return self.Gen.cNEGMaxUC_rule(m, xg, xt,
                                       self.connections['Generation'][xh])

    def cNEGMinUC_rule(self, m, xg, xt, xh):
        ''' Minimum generation '''
        return self.Gen.cNEGMinUC_rule(m, xg, xt,
                                       self.connections['Generation'][xh])

    def cNGenRampDown_rule(self, m, xg, xt, xh):
        ''' Generation ramps (down)'''
        return self.Gen.cNGenRampDown_rule(m, xg, xt, self.LLTime[xt],
                                           self.connections['Generation'][xh])

    def cNGenRampUp_rule(self, m, xg, xt, xh):
        ''' Generation ramps (up)'''
        return self.Gen.cNGenRampUp_rule(m, xg, xt, self.LLTime[xt],
                                         self.connections['Generation'][xh])

    def cNLDIni_rule(self, m, xt, xh):
        ''' Initialising dynamic loads '''
        return m.vNPump[self.connections['Pump'][xh], xt] == 0

    def cNLDMax_rule(self, m, xdl, xt, xh):
        ''' Maximum capacity of dynamic loads'''
        return (m.vNPump[self.connections['Pump'][xh]+xdl+1, xt] <=
                self.p['MaxPump'][xdl]/self.ENetwork.get_Base())

    def cNRESMax_rule(self, m, xt, xg, xh):
        ''' Maximum RES generation '''
        return (m.vNGen[self.resScenario[xg][xh][0], xt] <=
                self.scenarios['RES'][self.resScenario[xg][xh][1]+xt] *
                self.RES['Max'][xg])

    def cNServices_rule(self, m, xt, xh):
        ''' Provision of all services '''
        aux = xh*(self.generationE['Number']+1)+1
        return sum(m.vNGen[aux+x, xt] for x in self.s['GServices']) >= \
            sum(m.vNServ[xh*self.p['GServices']+xs, xt]
                for xs in range(self.p['GServices']))

    def cNServicesA_rule(self, m, xt, xh):
        ''' Provision of ancillary services '''
        aux = self.connections['Generation'][xh]
        return sum(m.vNGen[aux+x, xt] for x in self.Gen.get_GenAll()) >= \
            m.vNServ[xh*self.p['GServices'], xt]

    def cNServicesR_rule(self, m, xt, xh):
        ''' Provision of RES support services '''
        aux = self.connections['Generation'][xh]
        return sum(m.vNGen[aux+x, xt] for x in self.Gen.get_GenAll()) >= \
            m.vNServ[self.p['GServices']*(xh+1)-1, xt]

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

    def cNUncRES_rule(self, m, xt, xh):
        ''' Corrected maximum RES generation '''
        # TODO RES['Uncertainty'] is now a list
        return sum(m.vNGen[self.resScenario[xg][xh][0], xt]
                   for xg in self.s['RES']) <= \
            sum(self.scenarios['RES'][self.resScenario[xg][xh][1]+xt] *
                self.RES['Max'][xg] for xg in self.s['RES']) * \
            (1-self.RES['Uncertainty'][0])+m.vNFea[xh*self.p['faux']+1, xt] + \
            m.vNServ[self.p['GServices']*(xh+1)-1, xt]

    def cNUncRES0_rule(self, m, xt, xh):
        ''' Corrected maximum RES generation without support '''
        # TODO RES['Uncertainty'] is now a list
        return sum(m.vNGen[self.resScenario[xg][xh][0], xt]
                   for xg in self.s['RES']) <= \
            sum(self.scenarios['RES'][self.resScenario[xg][xh][1]+xt] *
                self.RES['Max'][xg] for xg in self.s['RES']) * \
            (1-self.RES['Uncertainty'][0])+m.vNFea[xh*self.p['faux']+1, xt]

    def get_ConB(self):
        ''' Get connections between branches '''
        return self.connections['Branches']

    def get_ConC(self, x=':'):
        ''' Get connections between costs '''
        return self.connections['Cost'][x]

    def get_ConG(self, x=':'):
        ''' Get connections between generators '''
        return self.connections['Generation'][x]

    def get_ConL(self, x=':'):
        ''' Get connections between lossess constraints '''
        return self.connections['Loss'][x]

    def get_ConFea(self, x=':'):
        ''' Get connection for feasibility constraints '''
        return self.connections['Feasibility'][x]

    def get_ConP(self, x=':'):
        ''' Get connection for pump constraints '''
        return self.connections['Pump'][x]

    def get_ConS(self):
        ''' Get connections between nodes '''
        return self.connections['set']

    def In_From_EM(self, m, xh, xg):
        ''' Connecting  inputs from pyeneE (MW --> MW) '''
        aux = self.connections['Generation'][xh]+self.Gen.get_vNGenH(xg)
        return sum(m.vNGen[aux, xt]*self.scenarios['Weights'][xt]
                   for xt in self.s['Tim'])*self.ENetwork.get_Base()

    def initialise(self):
        ''' Initialize externally '''
        # Setting additional constraints (Security, losses and feasibilty)

        # Read network data
        self.Read()

        self.ProcessENet()

        (self.busData, self.busScenario,
         self.resScenario) = self.ProcessEDem(self.demandE)

        self.ProcessEGen()

        self.NoBuses = self.ENetwork.get_NoBus()*(1+self.NoSec2)
        self.NoBranch = self.ENetwork.get_NoBra() + \
            (self.ENetwork.get_NoBra()-1)*self.NoSec2
        self.LLStor = np.zeros((self.ENetwork.get_NoBus(),
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
        if self.conventional['Ramp'][0] is not None:
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
                    self.conventional['Ramp'][0]*self.p['GenMax'][xg]
        else:
            if self.hydropower['Ramp'][0] is not None:
                # Only hydropower
                aux = self.hydropower['Number']
                xh = 0
                self.s['GRamp'] = np.zeros(aux, dtype=int)
                self.p['GRamp'] = np.zeros(aux, dtype=float)
            else:
                self.s['GRamp'] = []
                self.p['GRamp'] = []

        if self.hydropower['Ramp'][0] is not None:
            for xg in range(self.hydropower['Number']):
                self.s['GRamp'][xh] = self.conventional['Number']+xg
                self.p['GRamp'][xh] = self.hydropower['Ramp'][xg] * \
                    self.hydropower['Max'][xg]/self.ENetwork.get_Base()
                xh += 1

        # Sets and parameters for modelling Ancillary service requirements
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
        if self.RES['Number'] > 0 and self.RES['Uncertainty'][0] is not None:
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
        return (sum((sum(m.vNGCost[self.connections['Cost'][xh]+xg, xt]
                         for xg in self.s['Gen']) +
                     sum(m.vNFea[self.connections['Feasibility'][xh]+xf, xt]
                         for xf in self.s['Fea'])*1000000) *
                    self.scenarios['Weights'][xt] for xt in self.s['Tim']) -
                sum(self.pumps['Value'][xdl]*self.ENetwork.get_Base() *
                    sum(m.vNPump[self.connections['Pump'][xh] +
                                 xdl+1, xt]*self.scenarios['Weights'][xt]
                        for xt in self.s['Tim']) for xdl in self.s['Pump']))

    def offPrint(self):
        ''' Switch off the print flags '''
        for pars in self.Print.keys():
            self.Print[pars] = False

    def print(self, m, sh=None):
        ''' Print results '''
        if sh is None:
            sh = self.s['Con']

        for xh in sh:
            print("\n% CASE:", xh)

            if self.Print['GenBus']:
                print('\nFlow_EGen_Bus=',
                      self.generationE['Data']['GEN_BUS'], ';')

            if self.Print['Generation']:
                print("\nFlow_EGen=[")
                for xn in range(self.generationE['Number']+1):
                    for x2 in self.s['Tim']:
                        aux = (m.vNGen[self.connections['Generation'][xh]+xn,
                                       x2].value *
                               self.ENetwork.get_Base())
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['UC']:
                print("\nBin_EGen=[")
                aux = 1
                for xn in range(self.generationE['Number']+1):
                    for x2 in self.s['Tim']:
                        if self.settings['UC']:
                            aux = (m.vNGen_Bin[self.connections['Generation']
                                   [xh]+xn, x2].value)
                        print("%2.0f " % aux, end='')
                    print()
                print("];")

            if self.Print['Flows'] and self.settings['Flag']:
                print("\nFlow_EPower=[")
                for xb in range(self.ENetwork.get_NoBra()):
                    for x2 in self.s['Tim']:
                        aux = (m.vNFlow[self.connections['Flow'][xh] +
                                        xb, x2].value *
                               self.ENetwork.get_Base())
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
                aux = 0
                print("\nEPower_Loss=[")
                for xb in range(self.ENetwork.get_NoBra()):
                    for xt in self.s['Tim']:
                        if self.settings['Losses']:
                            aux = m.vNLoss[self.connections['Loss'][xh]+xb,
                                           xt].value * \
                                self.ENetwork.get_Base()
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Curtailment']:
                print("\nPumps=[")
                for xdl in self.s['Pump']:
                    for xt in self.s['Tim']:
                        aux = m.vNPump[self.connections['Pump'][xh]+xdl+1,
                                       xt].value*self.ENetwork.get_Base()
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Feasibility']:
                print("\nFeas=[")
                for xf in self.s['Fea']:
                    for xt in self.s['Tim']:
                        aux = m.vNFea[self.connections['Feasibility'][xh]+xf,
                                      xt].value*self.ENetwork.get_Base()
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.Print['Services'] and len(self.s['GServices']) > 0:
                print("\nServ=[")
                for xs in range(self.p['GServices']):
                    for xt in self.s['Tim']:
                        aux = m.vNServ[self.p['GServices']*xh+xs,
                                       xt].value*self.ENetwork.get_Base()
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

    def ProcessEDem(self, ENetDem):
        ''' Process demand and generation parameters '''
        # Adjust demand profiles
        busData = np.zeros(self.ENetwork.get_NoBus(), dtype=float)
        for xn in range(self.ENetwork.get_NoBus()):
            busData[xn] = self.demandE['PD'][xn]/self.ENetwork.get_Base()
        # Auxiliar to find demand profiles
        busScenario = np.zeros((self.ENetwork.get_NoBus(),
                                self.scenarios['Number']), dtype=int)

        if self.scenarios['NoDem'] > 0:
            acu = 0
            for xs in range(self.scenarios['Number']):
                for xn in range(self.ENetwork.get_NoBus()):
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
        self.Gen.initialise(self.ENetwork, self.settings, self.conventional)

        GenMax = self.generationE['Data']['PMAX']
        GenMin = self.generationE['Data']['PMIN']

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
            if xval[pwNo[xg]-1] == 0:
                for xv in range(pwNo[xg]):
                    GenLCst[acu+xv][0] = 0
                    GenLCst[acu+xv][1] = 0
            else:
                for xv in range(pwNo[xg]):
                    GenLCst[acu+xv][0] = (yval[xv+1] -
                                          yval[xv]) / (xval[xv+1]-xval[xv])
                    GenLCst[acu+xv][1] = yval[xv]-xval[xv]*GenLCst[acu+xv][0]
            acu += pwNo[xg]
        self.p['LLGenC'] = LLGenC

        # Changing to pu
        for xg in range(self.generationE['Number']):
            GenMax[xg] /= self.ENetwork.get_Base()
            GenMin[xg] /= self.ENetwork.get_Base()
        for xc in range(NoGenC):
            GenLCst[xc][0] *= self.ENetwork.get_Base()
        self.p['GenMax'] = GenMax
        self.p['GenMin'] = GenMin
        self.p['GenLCst'] = GenLCst

        self.NoGenC = NoGenC

    def ProcessENet(self):
        ''' Process information for optimisation purposes '''

        # Initialise electricity network object
        self.ENetwork.initialise(self.settings)

        # Add security constraints
        if len(self.settings['Security']) == 0 and \
                self.settings['SecurityFlag']:
            self.settings['Security'] = \
                [x+1 for x in range(self.connections['Branches'])]

        # Add security considerations
        NoSec2 = len(self.settings['Security'])

        # Number of parameters required for simulating security
        aux = self.connections['Branches']
        NoSec1 = self.connections['Branches']*(1+NoSec2)-NoSec2
        # Auxiliaries for modelling security considerations
        # Position of the variables
        LLESec1 = np.zeros((NoSec1, 2), dtype=int)

        for xb in range(self.connections['Branches']):
            LLESec1[xb][0] = xb
        aux = self.connections['Branches']
        x0 = aux
        xacu = 0
        for xs in range(NoSec2):
            xacu += self.ENetwork.get_NoBus()
            for xb in range(self.connections['Branches']):
                if xb+1 != self.settings['Security'][xs]:
                    LLESec1[x0][:] = [xb, xacu]
                    x0 += 1
        self.NoSec1 = NoSec1
        self.NoSec2 = NoSec2
        self.p['LLESec1'] = LLESec1

        # Add piece-wise power losses estimation
        if self.settings['Losses']:
            # Auxiliar for the cuadratic function to model losses
            # Choosing points for the lines
            aux = self.connections['Loss_Param']

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

            self.Number_LossCon = Number_LossCon
            self.ENetwork.loss['A'] = Loss_Con1
            self.ENetwork.loss['B'] = Loss_Con2

        # Add LL for dynamic loads
        LLDL = np.zeros(self.ENetwork.get_NoBus(), dtype=int)
        for xdl in range(self.pumps['Number']):
            LLDL[self.pumps['Bus'][xdl]-1] = xdl+1
        self.p['LLPump'] = LLDL

        # Add LL for feasibility constraints (Nodes)
        LLFea = np.zeros(self.ENetwork.get_NoBus()+1, dtype=int)
        if self.settings['Feasibility']:
            NoFea = self.ENetwork.get_NoBus()+1
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

        GenNCost = np.array(mpc['gencost']['COST'], dtype=int)
        NoOGen = len(GenNCost)

        # Defining device classes
        from pyene.engines.pyeneD import ElectricityNetwork, Generators

        # Define network model
        # The definition and configuration methods are separated so, in
        # principle the classes can be manually configured
        self.ENetwork = ElectricityNetwork(mpc['NoBus'], mpc["NoBranch"])
        self.ENetwork.MPCconfigure(mpc)

        # Define generator model
        self.Gen = Generators(NoOGen, self.hydropower['Number'],
                              self.RES['Number'])
        self.Gen.MPCconfigure(mpc, self.conventional, self.hydropower,
                              self.RES)

        self.connections['Branches'] = mpc['NoBranch']
        self.demandE = {
                'PD': np.array(mpc['bus']['PD'], dtype=float),
                'QD': np.array(mpc['bus']['QD'], dtype=float),
                }

        # Gen generation nodes (to mesure it)
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
            self.scenarios['Links'] = np.ones(self.ENetwork.get_NoBus() *
                                              self.scenarios['Number'],
                                              dtype=int)
            acu = self.ENetwork.get_NoBus()
            for xs in range(self.scenarios['Number']-1):
                for xt in range(self.ENetwork.get_NoBus()):
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
                ENetGen['MBASE'][xg] = self.ENetwork.get_Base()
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
                ENetGen['MBASE'][xg] = self.ENetwork.get_Base()
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

    def set_ConS(self, val):
        ''' Set connections for nodes '''
        self.connections['set'] = val

    def set_ConF(self, val):
        ''' Set connections between branch flows '''
        self.connections['Flow'] = val

    def set_ConV(self, val):
        ''' Set connections between voltages '''
        self.connections['Voltage'] = val

    def set_ConL(self, val):
        ''' Set connections between branch Losses '''
        self.connections['Loss'] = val

    def set_ConG(self, val):
        ''' Set connections between generators '''
        self.connections['Generation'] = val

    def set_ConC(self, val):
        ''' Set connections between costs '''
        self.connections['Cost'] = val

    def set_ConP(self, val):
        ''' Set connections between pumps '''
        self.connections['Pump'] = val

    def set_ConFea(self, val):
        ''' Set connections between feasibility constraints '''
        self.connections['Feasibility'] = val
