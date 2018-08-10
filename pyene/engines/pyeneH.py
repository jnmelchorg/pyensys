# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:44:55 2018

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import numpy as np
import networkx as nx
from pyomo.core import Constraint, Var, NonNegativeReals


class pyeneHConfig:
    ''' Default settings used for this class '''
    def __init__(self):
        # Basic settings
        self.settings = {
                'File': None,  # TODO create input json file
                'NoTime': 24,  # Number of time steps
                'Weights': None,  # Weight of the time period
                'Feas': True,  # Feasibility constraints
                'Penalty': 10000,  # Penalty for feasibility constraints
                'seconds': 3600,  # Time resolution
                'M': 1000,  # Multiplier to  reduce magnitude of time/storage
                'In': [[0, 1, 600], [1, 3, 800]]  # Fixed water inputs
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
                'River': None,  # Starting point of the scenario (Rivers)
                'LinksF': [[0, 0]],  # Connect scen 0:Begin1:End
                'LinksT': [[0, 1]]  # To scen 0:Begin1:End
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

    def _BuildHNetwork(self):
        ''' Build network model '''
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
        LLN2B2 = np.zeros((self.networkH.number_of_nodes(), 4), dtype=int)

        x0 = 0  # Initial position (LLaux)
        x1 = 0  # Initial position (branches)
        for xb in range(self.rivers['Number']):
            auxNode = [self.rivers['From'][xb]-1,
                       self.rivers['To'][xb]-1]
            auxX = [3, 1]
            for x2 in range(2):
                # Get next position
                xpos = LLN2B2[auxNode[x2]][auxX[x2]]
                # Initialize if the position is available
                if xpos == 0:
                    LLN2B2[auxNode[x2]][auxX[x2]] = x0
                    LLN2B2[auxNode[x2]][auxX[x2]-1] = 1
                else:  # Search for next available position
                    while LLnext[xpos] != 0:
                        xpos = LLnext[xpos]
                    # Storing data position
                    LLnext[xpos] = x0
                    LLN2B2[auxNode[x2]][auxX[x2]-1] = \
                        LLN2B2[auxNode[x2]][auxX[x2]-1]+1
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
                xpos = LLN2B2[xn][x2+1]
                if LLN2B2[xn][x2] != 0:
                    # Get other positions is available
                    LLN2B2[xn][x2+1] = xacu
                    xacu += LLN2B2[xn][x2]
                    for x3 in range(LLN2B2[xn][x2]):
                        # Store data sequentially
                        LLN2B1[x0] = LLaux[xpos]
                        xpos = LLnext[xpos]
                        x0 = x0+1

        # Linked list for nodes sending water to different rivers
        LLNodWeight = np.zeros(self.nodes['Number'], dtype=int)
        acu1 = 0
        self.opt['NoShare'] = 0
        for xn in range(self.nodes['Number']):
            if LLN2B2[xn][2] > 1:
                LLNodWeight[acu1] = xn
                acu1 += 1
                self.opt['NoShare'] += LLN2B2[xn][2]-1
        if acu1 == 0:
            self.opt['LLShare1'] = []
            self.opt['LLShare2'] = []
        else:
            self.opt['LLShare1'] = np.zeros((self.opt['NoShare'], 2),
                                            dtype=int)
            self.opt['LLShare2'] = np.zeros(self.opt['NoShare'], dtype=float)
            xr = 0
            for xn in LLNodWeight[0:acu1]:
                for xb in range(LLN2B2[xn][2]-1):
                    self.opt['LLShare1'][xr][:] = \
                        [LLN2B1[LLN2B2[xn][3]+xb], LLN2B1[LLN2B2[xn][3]+xb+1]]
                    self.opt['LLShare2'][xr] = \
                        self.rivers['Share'][LLN2B1[LLN2B2[xn][3]+xb+1]] / \
                        self.rivers['Share'][LLN2B1[LLN2B2[xn][3]+xb]]

        self.connections['LLN2B1'] = LLN2B1
        self.connections['LLN2B2'] = LLN2B2

    def _BuildLLScenario(self):
        ''' Build LL to locate beginning of each scenario '''
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

    def _BuildParts(self):
        ''' Get number of branches (rivers) and their parts '''
        self.rivers['Number'] = len(self.rivers['From'])
        self.nodes['Number'] = max([max(self.rivers['From']),
                                    max(self.rivers['To'])])
        if len(self.rivers['Parts']) == 0:
            # Default value of one
            self.rivers['Parts'] = np.ones(self.rivers['Number'], dtype=int)
        elif len(self.rivers['Parts']) == 1:
            # preselected value
            aux = self.rivers['Parts'][0]
            self.rivers['Parts'] = np.zeros(self.rivers['Number'], dtype=int)
            for xp in range(self.rivers['Number']):
                self.rivers['Parts'][xp] = aux

        # Create new river buckets
        Noaux = sum(self.rivers['Parts'])
        if Noaux > self.rivers['Number']:
            RiverFrom = np.zeros(Noaux, dtype=int)
            RiverTo = np.zeros(Noaux, dtype=int)
            RiverShare = np.ones(Noaux, dtype=float)
            RiverLength = np.zeros(Noaux, dtype=float)
            RiverSlope = np.zeros(Noaux, dtype=float)
            RiverWidth = np.zeros(Noaux, dtype=float)
            RiverDepthMax = np.zeros(Noaux, dtype=float)
            RiverDepthMin = np.zeros(Noaux, dtype=float)
            RiverManning = np.zeros(Noaux, dtype=float)

            xn = self.nodes['Number']
            xb = 0
            for xr in range(self.rivers['Number']):
                auxL = self.rivers['Length'][xr]/self.rivers['Parts'][xr]
                RiverFrom[xb] = self.rivers['From'][xr]
                RiverShare[xb] = self.rivers['Share'][xr]
                RiverLength[xb] = auxL
                RiverSlope[xb] = self.rivers['Slope'][xr]
                RiverWidth[xb] = self.rivers['Width'][xr]
                RiverDepthMax[xb] = self.rivers['DepthMax'][xr]
                RiverDepthMin[xb] = self.rivers['DepthMin'][xr]
                RiverManning[xb] = self.rivers['Manning'][xr]
                if self.rivers['Parts'][xr] > 1:
                    aux = self.rivers['To'][xr]
                    for xp in range(self.rivers['Parts'][xr]-1):
                        xn += 1
                        RiverTo[xb] = xn
                        xb += 1
                        RiverFrom[xb] = xn
                        RiverLength[xb] = auxL
                        RiverSlope[xb] = self.rivers['Slope'][xr]
                        RiverWidth[xb] = self.rivers['Width'][xr]
                        RiverDepthMax[xb] = self.rivers['DepthMax'][xr]
                        RiverDepthMin[xb] = self.rivers['DepthMin'][xr]
                        RiverManning[xb] = self.rivers['Manning'][xr]
                RiverTo[xb] = self.rivers['To'][xr]
                xb += 1

            self.nodes['Number'] = xn
            self.rivers['Number'] = Noaux
            self.rivers['From'] = RiverFrom
            self.rivers['To'] = RiverTo
            self.rivers['Share'] = RiverShare
            self.rivers['Length'] = RiverLength
            self.rivers['Slope'] = RiverSlope
            self.rivers['Width'] = RiverWidth
            self.rivers['DepthMax'] = RiverDepthMax
            self.rivers['DepthMin'] = RiverDepthMin
            self.rivers['Manning'] = RiverManning

    def _Process(self):
        '''Process information for the optimisation'''
        s = self.rivers['Slope']
        t = self.settings['seconds']
        w = self.rivers['Width']
        n = self.rivers['Manning']
        L = self.rivers['Length']
        M = self.settings['M']
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
                V[xc][xm] = (L[xc]*d[xc]*w[xc])/M  # Water volume
                S[xc][xm] = d[xc]**(2/3)*s[xc]**(1/2)/n[xc]  # Water speed
                D[xc][xm] = min([t*V[xc][xm], L[xc]])  # Distance

        # Get linear functios for estimating flows
        # Flows as a function of volume
        QLinear = np.zeros((self.rivers['Number'], 2), dtype=float)
        # Output flows as a function of input flows
        FLinear = np.zeros((self.rivers['Number'], 3), dtype=float)
        for xc in range(self.rivers['Number']):
            # Assigning linear approximation
            QLinear[xc][0] = (Q[xc][0]-Q[xc][1])/(V[xc][0]-V[xc][1])
            QLinear[xc][1] = Q[xc][1]-QLinear[xc][0]*V[xc][1]

            # Time dependent constraints for downstream flows
            aux1 = min(S[xc][0]*t, L[xc])
            aux1 = (aux1-0.5*aux1**2/t/S[xc][0])/L[xc]
            aux2 = min(S[xc][1]*t, L[xc])
            aux2 = (aux2-0.5*aux2**2/t/S[xc][1])/L[xc]

            # Linear approximation
            FLinear[xc][0] = (Q[xc][0]*aux1+(1-aux1)*Q[xc][1]-Q[xc][1]*aux2 -
                              (1-aux2)*Q[xc][1])/(Q[xc][0]-Q[xc][1])
            FLinear[xc][1] = Q[xc][0]*aux1+(1-aux1)*Q[xc][1] - \
                FLinear[xc][0]*Q[xc][0]
            aux3 = (Q[xc][0]*aux1+(1-aux1)*Q[xc][0]-Q[xc][1]*aux2 -
                    (1-aux2)*Q[xc][0])/(Q[xc][0]-Q[xc][1])
            aux4 = Q[xc][0]*aux1+(1-aux1)*Q[xc][0]-aux3*Q[xc][0]

            # Correction for error
            FLinear[xc][2] = (aux3*Q[xc][0]+aux4 -
                              FLinear[xc][0]*Q[xc][0]-FLinear[xc][1]) / \
                (Q[xc][0]-Q[xc][1])

        self.opt['Qmax'] = Q[:, 0]
        self.opt['Qmin'] = Q[:, 1]
        self.opt['QLinear'] = QLinear
        self.opt['FLinear'] = FLinear

    def addCon(self, m):
        ''' Add pyomo constraints '''
        # Constraint on maximum flow upstream
        m.cHQmaxUp = Constraint(m.sHBra, m.sHTim, m.sHSce,
                                rule=self.cHQmaxUp_rule)
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
        # Linking SoC in different scenarios
        if self.opt['NoSoCLinks'] > 0:
            m.cHSoCLink = Constraint(m.sHNoSoCLinks, m.sHBra,
                                     rule=self.cHSoCLink_rule)
        # Time dependent constraints on downstream flows
        m.cHQdownTime = Constraint(m.sHBra, m.sHTim, m.sHSce,
                                   rule=self.cHQdownTime_rule)
        # Fixed inputs
        if self.opt['NoFxInput'] > 0:
            m.cHFixedInput = Constraint(m.sHFXIn, rule=self.cHFixedInput_rule)

        return m

    def addPar(self, m):
        ''' Adding pyomo parameters '''
        m.pHConRiver = self.connections['River']  # Connectivity branches
        m.pHConInNode = self.connections['InNode']  # Node inputs
        m.pHConNode = self.connections['Node']  # Connectivity nodes
        m.pHConOutNode = self.connections['OutNode']  # Node outputs
        m.pHDeltaT = self.settings['seconds']/self.settings['M']  # dt
        m.pHFeas = self.opt['Feas']  # Feasibility constraint
        m.pHFeasTime = self.opt['FeasTime']  # Feasibility constraint
        m.pHFLinear = self.opt['FLinear']  # Time dependent flow approximation
        m.pHFXInput = self.settings['In']  # Fixed input constraints
        m.pHLLInNode = self.opt['InLL']  # Connection Node input
        m.pHLLN2B1 = self.connections['LLN2B1']  # Conection node2Branch
        m.pHLLN2B2 = self.connections['LLN2B2']  # Conection node2Branch
        m.pHLLOutNode = self.opt['OutLL']  # Connection Node Output
        m.pHLLShare1 = self.opt['LLShare1']  # From node to several rivers
        m.pHLLShare2 = self.opt['LLShare2']  # From node to several rivers
        m.pHOptQmax = self.opt['Qmax']  # Maximum flow
        m.pHOptQmin = self.opt['Qmin']  # Minimum flow
        m.pHPenalty = self.settings['Penalty']  # Penalty for feasibility
        m.pHQLinear = self.opt['QLinear']  # Volume to flow
        m.pHSoCLinks = self.opt['SoCLinks']  # Storage connections (scenarios)

        return m

    def addSets(self, m):
        ''' Adding pyomo sets '''
        m.sHBra = range(self.rivers['Number'])
        m.sHFXIn = range(self.opt['NoFxInput'])
        m.sHInNod = range(self.nodes['OutNumber'])
        m.sHNod = range(self.nodes['Number'])
        m.sHNoSoCLinks = range(self.opt['NoSoCLinks'])
        m.sHOutNod = range(self.nodes['OutNumber'])
        m.sHTim = range(self.settings['NoTime'])
        m.sHTimP = range(self.settings['NoTime']+1)
        m.sHSce = range(self.connections['Number'])
        m.sHShare = range(self.opt['NoShare'])

        return m

    def addVars(self, m):
        ''' Adding pyomo varaibles '''
        auxr = range(self.connections['Number']*self.rivers['Number'])
        auxin = range(self.connections['Number']*self.nodes['InNumber'])
        auxout = range(self.connections['Number']*self.nodes['OutNumber'])

        m.vHdown = Var(auxr, m.sHTim, domain=NonNegativeReals)
        m.vHFeas = Var(range(self.opt['FeasNo']*self.connections['Number']),
                       range(self.opt['FeasNoTime']), domain=NonNegativeReals,
                       initialize=0.0)
        m.vHin = Var(auxin, m.sHTim, domain=NonNegativeReals)
        m.vHout = Var(auxout, m.sHTim, domain=NonNegativeReals)
        m.vHSoC = Var(auxr, m.sHTimP, domain=NonNegativeReals)
        m.vHup = Var(auxr, m.sHTim, domain=NonNegativeReals)

        return m

    def cHFixedInput_rule(self, m, xf):
        ''' Fix some water inputs'''
        return m.vHin[m.pHFXInput[xf][0], m.pHFXInput[xf][1]] == \
            m.pHFXInput[xf][2]

    def cHQdownTime_rule(self, m, xr, xt, xh):
        ''' Time dependent constraint on minimum downstream flows '''
        return m.vHdown[m.pHConRiver[xh]+xr, xt] == m.pHFLinear[xr][0] * \
            m.vHup[m.pHConRiver[xh]+xr, xt]+m.pHFLinear[xr][1] - \
            m.vHFeas[m.pHFeas[xr], m.pHFeasTime[xt]] + \
            (m.vHSoC[m.pHConRiver[xh]+xr, xt]*m.pHQLinear[xr, 0] +
             m.pHQLinear[xr, 1]-m.pHOptQmin[xr])*m.pHFLinear[xr][2]

    def cHSoCLink_rule(self, m, xs, xr):
        ''' Linking SoC in different scenarios '''
        return m.vHSoC[m.pHSoCLinks[xs, 0]+xr, m.pHSoCLinks[xs, 1]] == \
            m.vHSoC[m.pHSoCLinks[xs, 2]+xr, m.pHSoCLinks[xs, 3]]

    def cHQmaxTime_rule(self, m, xr, xt, xh):
        ''' Time dependent constraint on maximum downstream flows '''
        return m.vHdown[m.pHConRiver[xh]+xr, xt] <= m.pHFLinear[xr][2] * \
            m.vHup[m.pHConRiver[xh]+xr, xt]+m.pHFLinear[xr][3] + \
            m.vHFeas[m.pHFeas[xr], m.pHFeasTime[xt]] + \
            (m.vHSoC[m.pHConRiver[xh]+xr, xt]*m.pHQLinear[xr, 0] +
             m.pHQLinear[xr, 1]-m.pHOptQmax[xr])*m.pHFLinear[xr][4]

    def cHQmaxUp_rule(self, m, xr, xt, xh):
        ''' Constraint on maximum flow upstream'''
        return m.vHup[m.pHConRiver[xh]+xr, xt] <= m.pHOptQmax[xr] + \
            m.vHFeas[m.pHFeas[xr], m.pHFeasTime[xt]]

    def cHQminUp_rule(self, m, xr, xt, xh):
        ''' Constraint on minimum flow upstream '''
        return m.vHup[m.pHConRiver[xh]+xr, xt] >= m.pHOptQmin[xr] - \
            m.vHFeas[m.pHFeas[xr], m.pHFeasTime[xt]]

    def cHNodeBalance_rule(self, m, xn, xt, xh):
        ''' Nodal balance '''
        return sum(m.vHin[m.pHConInNode[xh]+m.pHLLInNode[xn, 1], xt]
                   for xb in range(m.pHLLInNode[xn, 0])) + \
            sum(m.vHdown[m.pHConRiver[xh]+m.pHLLN2B1[m.pHLLN2B2[xn, 1]+xd], xt]
                for xd in range(m.pHLLN2B2[xn, 0])) == \
            sum(m.vHup[m.pHConRiver[xh]+m.pHLLN2B1[m.pHLLN2B2[xn, 3]+xd], xt]
                for xd in range(m.pHLLN2B2[xn, 2])) +\
            sum(m.vHout[m.pHConOutNode[xh]+m.pHLLOutNode[xn, 1], xt]
                for xb in range(m.pHLLOutNode[xn, 0]))

    def cHRiverBalance_rule(self, m, xr, xt, xh):
        ''' River balance '''
        aux = m.pHConRiver[xh]+xr
        return m.vHdown[aux, xt]*m.pHDeltaT == m.vHup[aux, xt]*m.pHDeltaT + \
            m.vHSoC[aux, xt]-m.vHSoC[aux, xt+1]

    def cHWeights_rule(self, m, xw, xt, xh):
        ''' Sharing water from a node among several rivers '''
        return m.vHup[m.pHConRiver[xh]+m.pHLLShare1[xw, 0], xt] * \
            m.pHLLShare2[xw] == \
            m.vHup[m.pHConRiver[xh]+m.pHLLShare1[xw, 1], xt]

    def initialise(self):
        ''' Initialise engine '''

        # Get number of branches (rivers) and their parts
        self._BuildParts()

        # Build LL to locate beginning of each scenario
        self._BuildLLScenario()

        # Get settings for each branch (part)
        self._Process()

        # Build network model
        self._BuildHNetwork()

        # List to connect the water available in the river at different times
        # e.g., volume at the end and beginning of scenario are the same
        if len(self.connections['LinksF']) > 0:
            aux = np.shape(self.connections['LinksF'])
            if len(aux) == 1:
                NoSoCLinks = 1
                val1 = np.zeros((1, 2), dtype=int)
                val2 = np.zeros((1, 2), dtype=int)
                val1[0][:] = self.connections['LinksF']
                val2[0][:] = self.connections['LinksT']
            else:
                NoSoCLinks = aux[0]
                val1 = self.connections['LinksF']
                val2 = self.connections['LinksT']

            SoCLinks = np.zeros((NoSoCLinks, 4), dtype=int)
            self.connections['River']
            aux = [0, self.settings['NoTime']]

            for xL in range(NoSoCLinks):
                SoCLinks[xL][:] = [self.connections['River'][val1[xL][0]],
                                   aux[val1[xL][1]],
                                   self.connections['River'][val2[xL][0]],
                                   aux[val2[xL][1]]]

        self.opt['SoCLinks'] = SoCLinks
        self.opt['NoSoCLinks'] = NoSoCLinks
        self.opt['NoFxInput'] = len(self.settings['In'])

    def OF_rule(self, m):
        ''' Objective function '''
        return sum(sum(m.vHout[xn, xt] for xn in m.sHInNod) +
                   sum(m.vHFeas[m.pHFeas[xr], m.pHFeasTime[xt]]
                   for xr in m.sHBra)*m.pHPenalty
                   for xt in m.sHTim)

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

    def print_settings(self, m):
        ''' Display input data and results in their original format '''
        auxr = range(self.connections['Number']*self.rivers['Number'])
        auxin = range(self.connections['Number']*self.nodes['InNumber'])
        auxout = range(self.connections['Number']*self.nodes['OutNumber'])

        print('vHSoC = [')
        for xr in auxr:
            for xt in m.sHTimP:
                print('%f ' % m.vHSoC[xr, xt].value, end='')
            print()
        print('];')

        print('vHup = [')
        for xr in auxr:
            for xt in m.sHTim:
                print('%f ' % m.vHup[xr, xt].value, end='')
            print()
        print('];')

        print('vHdown = [')
        for xr in auxr:
            for xt in m.sHTim:
                print('%f ' % m.vHdown[xr, xt].value, end='')
            print()
        print('];')

        print('vHin = [')
        for xn in auxin:
            for xt in m.sHTim:
                print('%f ' % m.vHin[xn, xt].value, end='')
            print()
        print('];')

        print('vHout = [')
        for xn in auxout:
            for xt in m.sHTim:
                print('%f ' % m.vHout[xn, xt].value, end='')
            print()
        print('];')

        print('vHFeas = [')
        for xn in range(self.opt['FeasNo']*self.connections['Number']):
            for xt in range(self.opt['FeasNoTime']):
                print('%f ' % m.vHFeas[xn, xt].value, end='')
            print()
        print('];')

        print('pHConRiver = ', self.connections['River'], end=';\n')
        print('pHConInNode = ', self.connections['InNode'], end=';\n')
        print('pHConOutNode = ', self.connections['OutNode'], end=';\n')
        print('pHLLN2B1 = ', self.connections['LLN2B1'], end=';\n')

        print('pHLLN2B2 = [')
        for xn in range(self.nodes['Number']):
            for x in range(4):
                print('%d ' % self.connections['LLN2B2'][xn][x], end='')
            print()
        print('];')

        print('pHLLInNode = [')
        for xn in range(self.nodes['Number']):
            for x in range(2):
                print('%d ' % self.opt['InLL'][xn][x], end='')
            print()
        print('];')

        print('pHLLOutNode = [')
        for xn in range(self.nodes['Number']):
            for x in range(2):
                print('%d ' % self.opt['OutLL'][xn][x], end='')
            print()
        print('];')

        print('pHOptQmax = ', self.opt['Qmax'], end=';\n')
        print('pHOptQmin = ', self.opt['Qmin'], end=';\n')
        print('pHFeas =', self.opt['Feas'], end=';\n')
        print('pHFeasTime = ', self.opt['FeasTime'], end=';\n')
        print('pHPenalty =', self.settings['Penalty'], end=';\n')
        print('pHLLShare1 =', self.opt['LLShare1'], end=';\n')
        print('pHLLShare2 =', self.opt['LLShare2'], end=';\n')

        print('pHQLinear =[')
        for xr in range(self.rivers['Number']):
            for x in range(2):
                print('%f ' % self.opt['QLinear'][xr][x], end='')
            print()
        print('];')

        print('pHDeltaT =', self.settings['seconds']/self.settings['M'],
              end=';\n')

        print('pHSoCLinks =[')
        for xs in range(self.opt['NoSoCLinks']):
            for x in range(4):
                print('%d ' % self.opt['SoCLinks'][xs][x], end='')
            print()
        print('];')

        print('pHFLinear =[')
        for xr in range(self.rivers['Number']):
            for x in range(3):
                print('%f ' % self.opt['FLinear'][xr][x], end='')
            print()
        print('];')
