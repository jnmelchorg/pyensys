# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:44:55 2018

@author: Dr Eduardo Alejandro Martínez Ceseña
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import math


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
                'set': range(5),  # Connections between nodes
                'From': [1, 2, 3, 4, 4],  # Node - from
                'To': [3, 3, 4, 5, 6],  # Node -to
                'Length': [1000, 1000, 1000, 1000, 1000],  # length (m)
                'Slope': [0.0001, 0.0001, 0.0001, 0.0001, 0.0001],  # Slope (m)
                'Width': [200, 200, 200, 200, 200],  # width (m)
                'Depth': [5, 5, 5, 5, 5],  # Maximum depth
                'Manning': [0.3, 0.3, 0.3, 0.3, 0.3],  # Mannings 'n
                'Links': [[1, 2, 1], [2, 3, 1], [3, 4, 0.5], [3, 5, 0.5]]
                }
        # Nodes
        self.Nodes = {
                'Number': 6,  # Number of scenarios
                'Links': None,  # Links between rivers
                'NoDem': 0,  # Number of demand profiles
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

    def Colebrook(self, err, Re):
        '''Colebrook equation'''
        f = 0.01
        dt = 0.00001
        f1 = 1000000000
        f2 = f
        cou = 0
        while abs(f1-f2) >= dt:
            cou += 1
            aux1 = -2*math.log(err/3.7+2.5/Re/math.sqrt(f))-1/math.sqrt(f)
            aux2 = (-2*math.log(err/3.7+2.5/Re/math.sqrt(f+dt)) -
                    1/math.sqrt(f+dt))
            aux2 = (aux2-aux1)/dt
            f = max(f-aux1/aux2, dt)
            f1 = f2
            f2 = f

            if cou >= 100:
                TypeError('Could not solve the Colebrook equation')

        return f
