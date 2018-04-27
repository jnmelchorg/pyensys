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
from pyeeN import ENetworkClass as dn
from pyeeB import EnergyClass as de


class pyeeClass():
# Get mathematical model
    def SingleLP(self, ENM):
        # Define pyomo model
        mod = ConcreteModel()

        #                                 Sets                                #
        mod = ENM.getSets(mod)

        #                              Parameters                             #
        mod = ENM.getPar(mod)

        #                           Model Variables                           #
        mod = ENM.getVars(mod)

        #                          Objective function                         #
        mod.OF = Objective(rule=ENM.OF_rule, sense=minimize)

        #                             Constraints                             #
        mod = ENM.addCon(mod)

        return mod

    # Network only optimisation
    def NSim(self, FileName):
        # Get network object
        NM = dn()

        # Initialise
        NM.initialise(FileName)

        # Build LP model
        NModel = self.SingleLP(NM)

        # Optimise
        opt = SolverFactory('glpk')

        # Print results
        results = opt.solve(NModel)
        NM.print(NModel)

    # Energy only optimisation
    def ESim(self, FileName):
        # Get energy object
        EM = de()

        # Initialise
        EM.initialise(FileName)

        # Build LP model
        EModel = self.SingleLP(EM)

        # Optimise
        opt = SolverFactory('glpk')

        # Print
        results = opt.solve(EModel)
        EM.print(EModel)

    # Energy and network simulation
    def ENSim(self, FileNameE, FileNameN):
        # Declare pyomo model
        mod = ConcreteModel()

        # Get Energy model
        EM = de()
        EM.initialise(FileNameE)
        EModel = self.SingleLP(EM)
        
        # Get size of network model
        NM = dn()
        NM.initialise(FileNameN)
        
        # Get number of required network model copies
        NoNM = 1+EM.NodeTime[EM.s_LL_time][1] - EM.NodeTime[EM.s_LL_time][0]

        # Settings for copying the network model
        NM.h = range(NoNM)
        NM.hFE = np.zeros(NoNM)
        NM.hVA = np.zeros(NoNM)
        NM.hEL = np.zeros(NoNM)
        NM.hFB = np.zeros(NoNM)
        NM.hFN = np.zeros(NoNM)
        NM.hG = np.zeros(NoNM)
        NM.hGC = np.zeros(NoNM)
        # Location of each copy
        for xc in NM.h:
            NM.hFE[xc] = int(xc*(NM.NoBranch+1))
            NM.hVA[xc] = int(xc*(NM.NoBuses+1))
            NM.hEL[xc] = int(xc*(NM.ENet.number_of_edges()+1))
            NM.hFB[xc] = int(xc*NM.ENet.number_of_edges())
            NM.hFN[xc] = int(xc*NM.ENet.number_of_nodes())
            NM.hG[xc] = int(xc*(NM.NoGen+1))
            NM.hGC[xc] = int(xc*NM.NoGen)

        for xc in range(EM.NodeTime[EM.s_LL_time][0],
                        1+EM.NodeTime[EM.s_LL_time][1]):
            print(xc)

        #                                 Sets                                #
        mod = EM.getSets(mod)
        mod = NM.getSets(mod)

        #                              Parameters                             #
        mod = EM.getPar(mod)
        mod = NM.getPar(mod)

        #                           Model Variables                           #
        mod = EM.getVars(mod)
        mod = NM.getVars(mod)

        #                             Constraints                             #
        mod = EM.addCon(mod)
        mod = NM.addCon(mod)

        #                          Objective function                         #
        mod.OF = Objective(rule=NM.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')

        # Print
        results = opt.solve(mod)
        #NM.print(mod)
        


# Get object
EN = pyeeClass()
FileNameE = "InputsTree4Periods.json"
FileNameN = "case4.json"

# Energy simulation
EN.ESim(FileNameE)
# Network simulation
EN.NSim(FileNameN)
# Joint simulation
EN.ENSim(FileNameE, FileNameN)