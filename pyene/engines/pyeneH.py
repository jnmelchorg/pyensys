# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:44:55 2018

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import math
import numpy as np
import networkx as nx
from pyomo.core import Constraint, Var, NonNegativeReals, Reals


class pyeneHConfig:
    ''' Default settings used for this class '''
    def __init__(self):
        # Basic settings
        self.settings = {
                'NoTime': 5,  # Number of time steps
                'Weights': None,  # Weight of the time period
                'Feas': True,  # Feasibility constraints
                'Penalty': 10000,  # Penalty for feasibility constraints
                'seconds': 3600  # Time resolution
                }
        # River models
        self.rivers = {
                'Number': None,  # Connections between nodes
                'From': [1, 2, 4, 4],  # Node - from
                'To': [2, 3, 5, 6],  # Node -to
                'Share': [1, 1, 0.4, 0.6],  # Links between water flows
                'Parts': [],
                'Length': [1000, 1000, 1000, 1000],  # length (m)
                'Slope': [0.0001, 0.0001, 0.0001, 0.0001],  # Slope (m)
                'Width': [200, 200, 200, 200],  # width (m)
                'DepthMax': [4, 4, 4, 4],  # Maximum depth
                'DepthMin': [1, 1, 1, 1],  # MInimum depth
                'Manning': [0.03, 0.03, 0.03, 0.03]  # Mannings 'n
                }
        # Connections between scenarios
        self.connections = {
                'Number': 1,  # Number of scenarios
                'set': None,  # set for scenarios
                'Node': None,  # Starting point of the scenario (Nodes)
                'InNode': None,  # Starting point of the scenario (Inputs)
                'OutNode': None,  # Starting point of the scenario (Outputs)
                'River': None  # Starting point of the scenario (Rivers)
                }
        # Nodes
        self.nodes = {
                'Number': None,  # Number of Nodes
                'In': [1, 4],  # Nodes with water inflows
                'Allowance': [1000, 1000],  # Water allowance
                'Out': [3, 5, 6]  # Nodes with water outflows
                }
        # Hydropower
        self.hydropower = {
                'Number': 0,  # Number of hydropower plants
                'Bus': [0],  # Location (Bus) in the network
                'Max': [0],  # Capacity (kW)
                'Cost': [0],  # Costs
                'Link': None  # Position of hydropower plants
                }
        # Optimisation
        self.opt = {
                'QLinear': None,
                'QMax': None,  # Maximum flow
                'QMin': None,  # Minimum flow
                'LL': None  # Linked lists
                }


class HydrologyClass:
    def __init__(self, obj=None):
        ''' Initialise network class '''
        # Get default values
        if obj is None:
            obj = pyeneHConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

    def _Process(self):
        '''Process information for the optimisation'''
        s = self.rivers['Slope']
        t = self.settings['seconds']
        w = self.rivers['Width']
        n = self.rivers['Manning']
        L = self.rivers['Length']
        P = self.rivers['Parts']
        Q = np.zeros((self.rivers['Number'], 2), dtype=float)
        V = np.zeros((self.rivers['Number'], 2), dtype=float)
        S = np.zeros((self.rivers['Number'], 2), dtype=float)
        D = np.zeros((self.rivers['Number'], 2), dtype=float)

        # Calculate additional parameters
        txt1 = ['DepthMax', 'DepthMin']
        for xm in range(2):  # Minimum and maximum values
            d = self.rivers[txt1[xm]]
            for xc in range(self.rivers['Number']):
                Q[xc][xm] = w[xc]*d[xc]**(5/3)*s[xc]**0.5/n[xc]  # Flow
                V[xc][xm] = L[xc]/P[xc]*d[xc]*w[xc]  # Water volume
                S[xc][xm] = d[xc]**(2/3)*s[xc]**(1/2)/n[xc]  # Water speed
                D[xc][xm] = min([t*V[xc][xm], L[xc]/P[xc]])  # Distance

        # Get linear functios for estimating flows
        # Flows as a function of volume
        QLinear = np.zeros((self.rivers['Number'], 2), dtype=float)
        # Output flows as a function of input flows
        FLinear = np.zeros((self.rivers['Number'], 2), dtype=float)
        for xc in range(self.rivers['Number']):
            # Assigning linear approximation
            QLinear[xc][0] = (Q[xc][0]-Q[xc][1])/(V[xc][0]-V[xc][1])
            QLinear[xc][1] = Q[xc][1]-QLinear[xc][0]*V[xc][1]

            # TODO Use time dependent constraints
            aux1 = (D[xc][0]-0.5*D[xc][0]**2/t/S[xc][0])/L[xc]*P[xc]
            aux2 = (D[xc][1]-0.5*D[xc][1]**2/t/S[xc][1])/L[xc]*P[xc]

        self.opt['Qmax'] = Q[:, 0]
        self.opt['Qmin'] = Q[:, 1]
        self.opt['QLinear'] = QLinear

    def addCon(self, m):
        ''' Add pyomo constraints '''
        # Constraint on maximum flow downstream
        m.cHQmaxDown = Constraint(m.sHBra, m.sHTim, m.sHSce,
                                  rule=self.cHQmaxDown_rule)
        # Constraint on maximum flow upstream
        m.cHQmaxUp = Constraint(m.sHBra, m.sHTim, m.sHSce,
                                rule=self.cHQmaxUp_rule)
        # Constraint on minimum flow downstream
        m.cHQminDown = Constraint(m.sHBra, m.sHTim, m.sHSce,
                                  rule=self.cHQminDown_rule)
        # Constraint on minimum flow upstream
        m.cHQminUp = Constraint(m.sHBra, m.sHTim, m.sHSce,
                                rule=self.cHQminUp_rule)
        # Nodal balance
        m.cHNodeBalance = Constraint(m.sHNod, m.sHTim, m.sHSce,
                                     rule=self.cHNodeBalance_rule)
        # River balance
        m.cHRiverBalance = Constraint(m.sHBra, m.sHTim, m.sHSce,
                                      rule=self.cHRiverBalance_rule)
        # Sharing water among connected rivers
        if self.opt['NoShare'] > 0:
            m.cHWeights = Constraint(m.sHShare, m.sHTim, m.sHSce,
                                     rule=self.cHWeights_rule)

#        # Nodal balance
#        m.Test0 = Constraint([0], m.sHTim, m.sHSce, rule=self.cHNodeBalance_rule)
#        
#        m.Test1 = Constraint(m.sHTim, rule=self.cTest1_rule)
#        m.Test2 = Constraint(m.sHTim, rule=self.cTest2_rule)
#        m.Test3 = Constraint(m.sHTim, rule=self.cTest3_rule)

        return m

    def addPar(self, m):
        ''' Adding pyomo parameters '''
        m.pHQLinear = self.opt['QLinear']
        m.pHConRiver = self.connections['River']
        m.pHConNode = self.connections['Node']
        m.pHConInNode = self.connections['InNode']
        m.pHConOutNode = self.connections['OutNode']
        m.pHLLN2B1 = self.connections['LLN2B1']
        m.pHLLN2B3 = self.connections['LLN2B3']
        m.pHLLInNode = self.opt['InLL']
        m.pHLLOutNode = self.opt['OutLL']
        m.pHOptQmax = self.opt['Qmax']
        m.pHOptQmin = self.opt['Qmin']
        m.pHFeas = self.opt['Feas']
        m.pHFeasTime = self.opt['FeasTime']
        m.pHPenalty = self.settings['Penalty']
        m.pHLLShare1 = self.opt['LLShare1']
        m.pHLLShare2 = self.opt['LLShare2']

        return m

    def addSets(self, m):
        ''' Adding pyomo sets '''
        m.sHNod = range(self.nodes['Number'])
        m.sHBra = range(self.rivers['Number'])
        m.sHTim = range(self.settings['NoTime'])
        m.sHTimP = range(self.settings['NoTime']+1)
        m.sHSce = range(self.connections['Number'])
        m.sHInNod = range(self.nodes['OutNumber'])
        m.sHOutNod = range(self.nodes['OutNumber'])
        m.sHShare = range(self.opt['NoShare'])

        return m

    def addVars(self, m):
        ''' Adding pyomo varaibles '''
        auxr = range(self.connections['Number']*self.rivers['Number'])
        auxin = range(self.connections['Number']*self.nodes['InNumber'])
        auxout = range(self.connections['Number']*self.nodes['OutNumber'])

        m.vHSoC = Var(auxr, m.sHTimP, domain=NonNegativeReals)
        m.vHup = Var(auxr, m.sHTim, domain=NonNegativeReals)
        m.vHdown = Var(auxr, m.sHTim, domain=NonNegativeReals)
        m.vHin = Var(auxin, m.sHTim, domain=NonNegativeReals)
        m.vHout = Var(auxout, m.sHTim, domain=NonNegativeReals)
        m.vHFeas = Var(range(self.opt['FeasNo']*self.connections['Number']),
                       range(self.opt['FeasNoTime']), domain=NonNegativeReals,
                       initialize=0.0)

        return m

    def cHQmaxDown_rule(self, m, xr, xt, xh):
        ''' Constraint on maximum flow downstream'''
        return m.vHdown[m.pHConRiver[xh]+xr, xt] <= m.pHOptQmax[xr] + \
            m.vHFeas[m.pHFeas[xr], m.pHFeasTime[xt]]

    def cHQmaxUp_rule(self, m, xr, xt, xh):
        ''' Constraint on maximum flow upstream'''
        return m.vHup[m.pHConRiver[xh]+xr, xt] <= m.pHOptQmax[xr] + \
            m.vHFeas[m.pHFeas[xr], m.pHFeasTime[xt]]

    def cHQminDown_rule(self, m, xr, xt, xh):
        ''' Constraint on minimum flow downstream'''
        return m.vHdown[m.pHConRiver[xh]+xr, xt] >= m.pHOptQmin[xr] - \
            m.vHFeas[m.pHFeas[xr], m.pHFeasTime[xt]]

    def cHQminUp_rule(self, m, xr, xt, xh):
        ''' Constraint on minimum flow upstream '''
        return m.vHup[m.pHConRiver[xh]+xr, xt] >= m.pHOptQmin[xr] - \
            m.vHFeas[m.pHFeas[xr], m.pHFeasTime[xt]]

    def cHNodeBalance_rule(self, m, xn, xt, xh):
        ''' Nodal balance '''
        return sum(m.vHin[m.pHConInNode[xh]+m.pHLLInNode[xn, 1], xt]
                   for xb in range(m.pHLLInNode[xn, 0])) + \
            sum(m.vHdown[m.pHConRiver[xh]+m.pHLLN2B1[m.pHLLN2B3[xn, 1]+xd], xt]
                for xd in range(m.pHLLN2B3[xn, 0])) == \
            sum(m.vHup[m.pHConRiver[xh]+m.pHLLN2B1[m.pHLLN2B3[xn, 3]+xd], xt]
                for xd in range(m.pHLLN2B3[xn, 2])) +\
            sum(m.vHout[m.pHConOutNode[xh]+m.pHLLOutNode[xn, 1], xt]
                for xb in range(m.pHLLOutNode[xn, 0]))

    def cHWeights_rule(self, m, xw, xt, xh):
        ''' Sharing water from a node among several rivers '''
        return m.vHup[m.pHConRiver[xh]+m.pHLLShare1[xw, 0], xt] * \
            m.pHLLShare2[xw] == \
            m.vHup[m.pHConRiver[xh]+m.pHLLShare1[xw, 1], xt]

    def cTest1_rule(self, m, xt):
        return m.vHin[0, xt] == m.vHup[0, xt]

    def cTest2_rule(self, m, xt):
        return m.vHdown[0, xt] == m.vHup[1, xt]

    def cTest3_rule(self, m, xt):
        return m.vHdown[1, xt] == m.vHout[0, xt]

    def cHRiverBalance_rule(self, m, xr, xt, xh):
        ''' River balance '''
        aux = m.pHConRiver[xh]+xr
        return m.vHdown[aux, xt] == m.vHup[aux, xt] + \
            m.vHSoC[aux, xt]-m.vHSoC[aux, xt+1]

    def OF_rule(self, m):
        ''' Objective function '''
        return sum(sum(m.vHout[xn, xt] for xn in m.sHInNod) +
                   sum(m.vHFeas[m.pHFeas[xr], m.pHFeasTime[xt]]
                   for xr in m.sHBra)*m.pHPenalty
                   for xt in m.sHTim)

    def initialise(self):
        ''' Initialise engine '''

        # Get number of branches (rivers) and their parts
        self.rivers['Number'] = len(self.rivers['From'])
        if len(self.rivers['Parts']) == 0:
            # Default value of one
            self.rivers['Parts'] = np.ones(self.rivers['Number'], dtype=int)
        elif len(self.rivers['Parts']) == 1:
            # preselected value
            aux = self.rivers['Parts'][0]
            self.rivers['Parts'] = np.zeros(self.rivers['Number'], dtype=int)
            for xp in range(self.rivers['Number']):
                self.rivers['Parts'][xp] = aux

        # Build LL to connect branches with their parts
        Noaux = sum(self.rivers['Parts'])
        self.opt['LL'] = np.ones(Noaux, dtype=int)
        x = 0
        for xL in range(Noaux):
            self.opt['LL'][x] = xL
            x = x+1

        # Build LL to locate beginning of each scenario
        self.nodes['Number'] = max([max(self.rivers['From']),
                                    max(self.rivers['To'])])
        self.nodes['InNumber'] = len(self.nodes['In'])
        self.nodes['OutNumber'] = len(self.nodes['Out'])
        self.connections['set'] = range(self.connections['Number'])

        self.connections['Node'] = np.zeros(self.connections['Number'],
                                            dtype=int)
        self.connections['River'] = np.zeros(self.connections['Number'],
                                             dtype=int)
        self.connections['InNode'] = np.zeros(self.connections['Number'],
                                              dtype=int)
        self.connections['OutNode'] = np.zeros(self.connections['Number'],
                                               dtype=int)
        for xh in self.connections['set']:
            self.connections['Node'][xh] = self.nodes['Number']*xh
            self.connections['River'][xh] = self.rivers['Number']*xh
            self.connections['InNode'][xh] = self.nodes['InNumber']*xh
            self.connections['OutNode'][xh] = self.nodes['OutNumber']*xh

        # Build LL to flag positions of input.output nodes
        self.opt['InLL'] = np.zeros((self.nodes['Number'], 2), dtype=int)
        self.opt['OutLL'] = np.zeros((self.nodes['Number'], 2), dtype=int)
        for xn in range(self.nodes['InNumber']):
            self.opt['InLL'][self.nodes['In'][xn]-1][:] = [1, xn]
        for xn in range(self.nodes['OutNumber']):
            self.opt['OutLL'][self.nodes['Out'][xn]-1][:] = [1, xn]

        # Add feasibility constraints
        self.opt['Feas'] = np.zeros(self.rivers['Number'], dtype=int)
        self.opt['FeasTime'] = np.zeros(self.settings['NoTime'], dtype=int)
        if self.settings['Feas']:
            self.opt['FeasNo'] = self.rivers['Number']
            self.opt['FeasNoTime'] = self.settings['NoTime']
            for xr in range(self.opt['FeasNo']):
                self.opt['Feas'][xr] = xr
            for xt in range(self.opt['FeasNoTime']):
                self.opt['FeasTime'][xt] = xt
        else:
            self.opt['FeasNo'] = 1
            self.opt['FeasNoTime'] = 1

        # Get settings for each branch (part)
        self._Process()

        # Build network model
        self.networkH = nx.Graph()
        # Adding nodes
        aux = min([min(self.rivers['From']), min(self.rivers['To'])])
        for xn in range(self.nodes['Number']):
            self.networkH.add_node(aux+xn)
        # Adding branches
        for xb in range(self.rivers['Number']):
            self.networkH.add_edge(self.rivers['From'][xb],
                                   self.rivers['To'][xb])

        # Map connections between nodes and branches (non-sequential search)
        NoN2B = self.networkH.number_of_edges()*2  # Number of data points
        LLaux = np.zeros(NoN2B, dtype=int)  # connections (non-sequential)
        LLnext = np.zeros(NoN2B, dtype=int)  # Next connection (non-sequential)
        LLN2B1 = np.zeros(NoN2B, dtype=int)  # connections (sequential)
        # Position of first connection and number of connections
        LLN2B3 = np.zeros((self.networkH.number_of_nodes(), 4), dtype=int)

        x0 = 0  # Initial position (LLaux)
        x1 = 0  # Initial position (branches)
        for xb in range(self.rivers['Number']):
            auxNode = [self.rivers['From'][xb]-1,
                       self.rivers['To'][xb]-1]
            auxX = [3, 1]
            for x2 in range(2):
                # Get next position
                xpos = LLN2B3[auxNode[x2]][auxX[x2]]
                # Initialize if the position is available
                if xpos == 0:
                    LLN2B3[auxNode[x2]][auxX[x2]] = x0
                    LLN2B3[auxNode[x2]][auxX[x2]-1] = 1
                else:  # Search for next available position
                    while LLnext[xpos] != 0:
                        xpos = LLnext[xpos]
                    # Storing data position
                    LLnext[xpos] = x0
                    LLN2B3[auxNode[x2]][auxX[x2]-1] = \
                        LLN2B3[auxNode[x2]][auxX[x2]-1]+1
                # Storing data point
                LLaux[x0] = x1
                x0 += 1
            x1 += 1

        # Remove the 'next' by arranging the data sequentially
        x0 = 0  # Position LLN2B1
        xacu = 0  # Total number of positions addressed so far
        for x2 in [2, 0]:
            for xn in range(self.networkH.number_of_nodes()):
                # Get first branch position for this node
                xpos = LLN2B3[xn][x2+1]
                if LLN2B3[xn][x2] != 0:
                    # Get other positions is available
                    LLN2B3[xn][x2+1] = xacu
                    xacu += LLN2B3[xn][x2]
                    for x3 in range(LLN2B3[xn][x2]):
                        # Store data sequentially
                        LLN2B1[x0] = LLaux[xpos]
                        xpos = LLnext[xpos]
                        x0 = x0+1

        # Linked list for nodes sending water to different rivers
        LLNodWeight = np.zeros(self.nodes['Number'], dtype=int)
        acu1 = 0
        self.opt['NoShare'] = 0
        for xn in range(self.nodes['Number']):
            if LLN2B3[xn][2] > 1:
                LLNodWeight[acu1] = xn
                acu1 += 1
                self.opt['NoShare'] += LLN2B3[xn][2]-1
        if acu1 == 0:
            self.opt['LLShare1'] = []
            self.opt['LLShare2'] = []
        else:
            self.opt['LLShare1'] = np.zeros((self.opt['NoShare'], 2),
                                            dtype=int)
            self.opt['LLShare2'] = np.zeros(self.opt['NoShare'], dtype=float)
            xr = 0
            for xn in LLNodWeight[0:acu1]:
                for xb in range(LLN2B3[xn][2]-1):
                    self.opt['LLShare1'][xr][:] = \
                        [LLN2B1[LLN2B3[xn][3]+xb], LLN2B1[LLN2B3[xn][3]+xb+1]]
                    self.opt['LLShare2'][xr] = \
                        self.rivers['Share'][LLN2B1[LLN2B3[xn][3]+xb+1]] / \
                        self.rivers['Share'][LLN2B1[LLN2B3[xn][3]+xb]]

        self.connections['LLN2B1'] = LLN2B1
        self.connections['LLN2B3'] = LLN2B3

    def print(self, m):
        ''' Print results '''
        self.print = {
                'WIn': True,
                'WOut': True,
                'Fup': True,
                'Fdown': True,
                'SoC': True,
                'Feas': True
                }
        # Nodal results
        for xh in m.sHSce:
            print('\nCASE:', xh)

            if self.print['WIn']:
                print("\nWater_In_Node=[")
                for xn in m.sHNod:
                    for xt in m.sHTim:
                        if m.pHLLInNode[xn, 0] != 0:
                            aux = m.vHin[m.pHConInNode[xh] +
                                         m.pHLLInNode[xn, 1], xt].value
                        else:
                            aux = 0
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.print['WOut']:
                print("\nWater_Out_Node=[")
                for xn in m.sHNod:
                    for xt in m.sHTim:
                        if m.pHLLOutNode[xn, 0] != 0:
                            aux = m.vHout[m.pHConOutNode[xh] +
                                          m.pHLLOutNode[xn, 1], xt].value
                        else:
                            aux = 0
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.print['Fup']:
                print("\nFlow_Upstream=[")
                for xr in m.sHBra:
                    for xt in m.sHTim:
                        aux = m.vHup[m.pHConRiver[xh]+xr, xt].value
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.print['Fdown']:
                print("\nFlow_Downstream=[")
                for xr in m.sHBra:
                    for xt in m.sHTim:
                        aux = m.vHdown[m.pHConRiver[xh]+xr, xt].value
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.print['SoC']:
                print("\nRiver=[")
                for xr in m.sHBra:
                    for xt in m.sHTimP:
                        aux = m.vHSoC[m.pHConRiver[xh]+xr, xt].value
                        print("%8.4f " % aux, end='')
                    print()
                print("];")

            if self.print['Feas']:
                print("\nFlow_Feasibility=[")
                for xr in m.sHBra:
                    for xt in m.sHTim:
                        aux = m.vHFeas[m.pHFeas[xr], m.pHFeasTime[xt]].value
                        print("%8.4f " % aux, end='')
                    print()
                print("];")
