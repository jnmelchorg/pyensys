# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:44:55 2018

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import math
import numpy as np

class HNetworkClass:
    # Initialize
    def __init__(self):
        # Basic settings
        self.settings = {
                'NoTime': 5,  # Number of time steps
                'Weights': None,  # Weight of the time period
                'Scenarios': 1,  # Number of scenarios
                'Spill': True  # Feasibility constraints
                }
        # Connections
        self.connections = {
                'Number': 5,  # Connections between nodes
                'From': [1, 2, 3, 4, 4],  # Node - from
                'To': [3, 3, 4, 5, 6],  # Node -to
                'Length': [1000, 1000, 1000, 1000, 1000],  # length (m)
                'Slope': [0.0001, 0.0001, 0.0001, 0.0001, 0.0001],  # Slope (m)
                'Width': [200, 200, 200, 200, 200],  # width (m)
                'DepthMax': [5, 5, 5, 5, 5],  # Maximum depth
                'DepthMin': [1, 1, 1, 1, 1],  # MInimum depth
                'Manning': [0.03, 0.03, 0.03, 0.03, 0.03],  # Mannings 'n
                'Links': [[1, 2, 1], [2, 3, 1], [3, 4, 0.5], [3, 5, 0.5]],
                'QMax': None,  # Maximum flow
                'QMin': None,  # Minimum flow
                }
        # Nodes
        self.Nodes = {
                'Number': 6,  # Number of Nodes
                'Qallowance': 1000,  # Water allowance for the simulation
                'QIn': None,  # Water inputs per time period
                'QOut': 0,  # Water outputs per scenario
                'Demand': [1],  # Demand profiles
                'NoRES': 0,  # Number of RES profiles
                'LinksRes': 'Default',  # Links RES generators and profiles
                'RES': [0],  # Location of the RES profiles
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
        self.Opt = {
                'QLinear': None
                }

    def Read(self, FileName, jsonPath):
        '''Reading hydraulic network from file'''
        NotImplementedError('self.Read is to be created')

    def _Process(self):
        '''Process information for the optimisation'''
        Qmax = np.zeros(self.connections['Number'], dtype=float)
        Qmin = np.zeros(self.connections['Number'], dtype=float)
        s = self.connections['Slope']
        w = self.connections['Width']
        dmax = self.connections['DepthMax']
        dmin = self.connections['DepthMin']
        n = self.connections['Manning']
        L = self.connections['Length']
        QLinear = np.zeros((self.connections['Number'], 2), dtype=float)
        for xc in range(self.connections['Number']):
            # Getting range for flows
            Qmax[xc] = w[xc]*dmax[xc]**(5/3)*s[xc]**0.5/n[xc]
            Qmin[xc] = w[xc]*dmin[xc]**(5/3)*s[xc]**0.5/n[xc]
            # Getting volumes
            vmax = L[xc]*dmax[xc]*w[xc]
            vmin = L[xc]*dmin[xc]*w[xc]
            # Assigning linear approximation
            QLinear[xc][0] = (Qmax[xc]-Qmin[xc])/(vmax-vmin)
            QLinear[xc][1] = Qmin[xc]-QLinear[xc][0]*vmin

        self.connections['Qmax'] = Qmax
        self.connections['Qmin'] = Qmin
        self.Opt['QLinear'] = QLinear

HN = HNetworkClass()
HN._Process()
print(HN.Opt['QLinear'])
            
