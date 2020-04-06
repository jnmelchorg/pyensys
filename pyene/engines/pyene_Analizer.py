"""
Created on Mon April 06 2020

This python file containts the classes and methods for the analysis and
modifications of the topology and electrical characteristics of power system.
Furthermore, tools to build the temporal tree are provided

@author: Dr. Jose Nicolas Melchor Gutierrez
"""

import networkx as nx

class TemporalTree():
    
    G = nx.DiGraph()

class PowerSystemReduction():
    ''' This class contains all necessary methods to reduce a network depending 
    on the requirements of the user
    
    The options considered are:
    
    1. Simplify generators connected to the same node to an equivalent generator
    2. Simplify loads connected to the same node to an equivalent load
    2. Simplify power system network until a desired voltage level '''
    def Reduction(self):
        ''' This is the main class method'''
        self.Networkreduction()
    
    def Networkreduction(self):
        ''' This class method controls the network reduction '''
        G = nx.MultiGraph()