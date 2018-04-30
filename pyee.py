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

    #                           Objective function                            #
    def OF_rule(self, m): 
        return sum(sum(sum(m.vGCost[self.hGC[xh]+xg, xt] for xg in m.sGen)
                           for xt in m.sTim) +
                   1000000*sum(m.vFeaB[self.hFB[xh]+xb] for xb in m.sBra) +
                   1000000*sum(m.vFeaN[self.hFN[xh]+xn] for xn in m.sBus)
                   for xh in self.h)
    
    # Water consumption depends on water use by the electricity system
    def EMNM_rule(self, m, xL, xv):
        return (m.WOutFull[m.LLENM[xL][0], xv] == 
                sum(m.vGen[m.LLENM[xL][1]+xv, xt] for xt in m.sTim))

    #                               Constraints                               #
    def addCon(self, m):
        # Link water consumption from both models
        m.EMNM = Constraint(m.sLLEN, m.sVec, rule=self.EMNM_rule)

        return m
        
    #                                   Sets                                  #
    def getSets(self, m):
        m.sLLEN = range(self.NoLL)

        return m

    #                                Parameters                               #
    def getPar(self, m):
        m.LLENM = self.LLENM

        return m

    #                             Model Variables                             #
    def getVars(self, m):
        # Converting some parameters to variables
        del m.WOutFull
        m.WOutFull = Var(m.sNodz, m.sVec, domain=NonNegativeReals,
                         initialize=0.0)
        return m

    # Energy and network simulation
    def ENSim(self, FileNameE, FileNameN):
        # Declare pyomo model
        mod = ConcreteModel()

        # Degine Energy model and network models
        EM = de()
        NM = dn()

        # Get components of energy model
        EM.initialise(FileNameE)
        EModel = self.SingleLP(EM)

        # Add hydropower units to network model
        # CURRENLTY ASSIMUNC CHARACTERISTICS OF HYDRO
        NM.NoHyd = EM.NosVec
        NM.PosHyd = np.zeros(NM.NoHyd, dtype=int)
        NM.HydPMax = np.zeros(NM.NoHyd, dtype=int)
        NM.HydQMax = np.zeros(NM.NoHyd, dtype=int)
        NM.HydCst = np.zeros(NM.NoHyd, dtype=float)
        for xv in range(EM.NosVec):
            NM.PosHyd[xv] = xv+1
            NM.HydPMax[xv] = 1000
            NM.HydQMax[xv] = 1000
            NM.HydCst[xv] = 0.0001  # Assigning a small cost to hidropower

        # Get size of network model
        NM.initialise(FileNameN)
        
        # Get number of required network model copies
        NoNM = 1+EM.NodeTime[EM.s_LL_time][1] - EM.NodeTime[EM.s_LL_time][0]

        # Settings for copying the network model
        NM.h = range(NoNM)
        NM.hFE = np.zeros(NoNM, dtype=int)
        NM.hVA = np.zeros(NoNM, dtype=int)
        NM.hEL = np.zeros(NoNM, dtype=int)
        NM.hFB = np.zeros(NoNM, dtype=int)
        NM.hFN = np.zeros(NoNM, dtype=int)
        NM.hG = np.zeros(NoNM, dtype=int)
        NM.hGC = np.zeros(NoNM, dtype=int)
        # Location of each copy
        for xc in NM.h:
            NM.hFE[xc] = xc*(NM.NoBranch+1)
            NM.hVA[xc] = xc*(NM.NoBuses+1)
            NM.hEL[xc] = xc*(NM.ENet.number_of_edges()+1)
            NM.hFB[xc] = xc*NM.ENet.number_of_edges()
            NM.hFN[xc] = xc*NM.ENet.number_of_nodes()
            NM.hG[xc] = xc*(NM.NoGen+1)
            NM.hGC[xc] = xc*NM.NoGen

        # Build LL to link the models through hydro consumption
        self.NoLL = 1+EM.NodeTime[EM.s_LL_time][1]-EM.NodeTime[EM.s_LL_time][0]
        self.LLENM = np.zeros((self.NoLL, 2), dtype=int)
        for xc in range(self.NoLL):
            self.LLENM[xc][:] = [EM.NodeTime[EM.s_LL_time][0]+xc,
                                 NM.hG[xc]+NM.NoOGen+1]

        # Taking sets for modelling
        self.hGC = NM.hGC
        self.hFB = NM.hFB
        self.hFN = NM.hFN
        self.h = NM.h

        #                                 Sets                                #
        mod = EM.getSets(mod)
        mod = NM.getSets(mod)
        mod = self.getSets(mod)

        #                              Parameters                             #
        mod = EM.getPar(mod)
        mod = NM.getPar(mod)
        mod = self.getPar(mod)

        #                           Model Variables                           #
        mod = EM.getVars(mod)
        mod = NM.getVars(mod)
        mod = self.getVars(mod)

        #                             Constraints                             #
        mod = EM.addCon(mod)
        mod = NM.addCon(mod)
        mod = self.addCon(mod)

        #                          Objective function                         #
        mod.OF = Objective(rule=self.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')

        # Print
        results = opt.solve(mod)
        NM.print(mod)
        


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