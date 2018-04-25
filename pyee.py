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

        # Get connection nodes
        


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