# Python Energy & Network Engine
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
from pyeneN import ENetworkClass as dn  # Network component
from pyeneE import EnergyClass as de  # Energy balance/aggregation component


class pyeneClass():
    # Initialisation
    def __init__(self):
        # Chose to load data from file
        self.fRea = True

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

        #                          Objective function                         #
        NModel.OF = Objective(rule=NM.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')

        # Print results
        results = opt.solve(NModel)
        NM.print(NModel)

    # Energy only optimisation
    def ESim(self, FileName):
        # Get energy object
        EM = de()

        # Chose to load data from file
        EM.fRea = True

        # Initialise
        EM.initialise(FileName)

        # Build LP model
        EModel = self.SingleLP(EM)

        #                          Objective function                         #
        EModel.OF = Objective(rule=EM.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')

        # Print
        results = opt.solve(EModel)
        EM.print(EModel)

    #                           Objective function                            #
    def OF_rule(self, m):
        return sum(sum(sum(m.vGCost[self.hGC[xh]+xg, xt] for xg in m.sGen) +
                       sum(m.vFea[self.hFea[xh]+xf, xt] for xf in m.sFea) *
                       1000000 for xt in m.sTim) -
                   sum(m.ValDL[xdl]*sum(m.vGenDL[self.hDL[xh]+xdl+1, xt]
                                        for xt in m.sTim) for xdl in m.sDL)
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

    # Initialise energy and network simulation
    def Initialise_ENSim(self, EM, NM, FileNameE,FileNameN):
        # Get components of energy model
        if self.fRea == True:
            # Chose to load data from file
            EM.fRea = True
        else:
            FileNameE = "NoName"

        EM.initialise(FileNameE)

        # Add hydropower units to network model
        # CURRENLTY ASSIMUNG CHARACTERISTICS OF HYDRO
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
        NM.hDL = np.zeros(NoNM, dtype=int)
        NM.hFea = np.zeros(NoNM, dtype=int)
        # Location of each copy
        for xc in NM.h:
            NM.hFE[xc] = xc*(NM.NoBranch+1)
            NM.hVA[xc] = xc*(NM.NoBuses+1)
            NM.hEL[xc] = xc*(NM.ENet.number_of_edges()+1)
            NM.hFB[xc] = xc*NM.ENet.number_of_edges()
            NM.hFN[xc] = xc*NM.ENet.number_of_nodes()
            NM.hG[xc] = xc*(NM.NoGen+1)
            NM.hGC[xc] = xc*NM.NoGen
            NM.hDL[xc] = xc*(NM.NoDL+1)
            NM.hFea[xc] = xc*NM.NoFea

        # Build LL to link the models through hydro consumption
        self.NoLL = 1+EM.NodeTime[EM.s_LL_time][1]-EM.NodeTime[EM.s_LL_time][0]
        self.LLENM = np.zeros((self.NoLL, 2), dtype=int)
        for xc in range(self.NoLL):
            self.LLENM[xc][:] = [EM.NodeTime[EM.s_LL_time][0]+xc,
                                 NM.hG[xc]+NM.NoOGen+1]

        # Taking sets for modelling local objective function
        self.hGC = NM.hGC
        self.hFB = NM.hFB
        self.hFN = NM.hFN
        self.hDL = NM.hDL
        self.hFea = NM.hFea
        self.h = NM.h

        return (EM, NM)

    # Build pyomo model
    def build_Mod(self, EM, NM):
        # Declare pyomo model
        mod = ConcreteModel()
        mod = self.SingleLP(EM)
        
        #                                 Sets                                #
        mod = EM.getSets(mod)
        mod = NM.getSets(mod)
        mod = self.getSets(mod)

         #                           Model Variables                           #
        mod = EM.getVars(mod)
        mod = NM.getVars(mod)

        #                              Parameters                             #
        mod = EM.getPar(mod)
        mod = NM.getPar(mod)
        mod = self.getPar(mod)

        return mod


    def add_Hydro(self, mod, HydroIn):
        aux=np.array(HydroIn)
        if aux.size == 1:
            mod.WInFull[1] = HydroIn
        else:
            mod.WInFull[1][:] = HydroIn

        return mod

    # Run pyomo model
    def Run_Mod(self, mod, EM, NM):
        # Finalise model
        #                      Model additional variables                     #
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

        return mod

    # Run Energy and network combined model
    def Run_ENSim(self, mod, EM, NM, HydroIn):
        # Build model
        mod = build_Mod(EM, NM)
        # Add water
        mod = add_Hydro(mod, HydroIn)
        # Run model
        Run_Mod(mod, EM, NM)
        

    # Run Energy and network combined model
    def Print_ENSim(self, mod, EM, NM):
        # Print results
        EM.print(mod)        
        NM.print(mod)
        print('Water outputs:')
        for xn in mod.sNodz:
            for xv in mod.sVec:
                aux = mod.WOutFull[xn, xv].value
                print("%8.4f " % aux, end='')
            print('')
        print('Water inputs:\n', mod.WInFull)

    # Print initial assumptions
    def _CheckInE(self, EM):
        if EM.fRea:
            print('Loading resolution tree from file')
        else:
            print('Predefined resolution tree')
            print(EM.Input_Data)

    # Print initial assumptions
    def _CheckInN(self, NM):        
        if NM.Add_Loss:
            print('Losses are considered')
        else:
            print('Losses are neglected')        
        if NM.Add_Fea:
            print('Feasibility constrants are included')
        else:
            print('Feasibility constraints are neglected')
        print('Demand multiplyiers ', NM.DemProf)
        print('Secuity: ', NM.Sec)

    # Run tests
    def _runTests(self, EN):
        # Get object        
        FileNameE = "ResolutionTreeMonth01.json"
        FileNameN = "case4.json"

        # Energy simulation
        print('\n\nTESTING ENERGY BALANCE AND AGGREGATION\n\n')
        EN.ESim(FileNameE)
        # Network simulation
        print('\n\nTESTING NETWORK MODELS\n\n')
        EN.NSim(FileNameN)
        # Joint simulation
        print('\n\nTESTING INTEGRATED MODEL\n\n')
        # Creat objects
        EM = de()
        NM = dn()
        # Change assumptions
        NM.DemProf = [1, 1.1, 1.2]#[0.5, 0.4, 0.5, 0.4]
        # NM.DemProf = [1, 1.1]
        NM.Add_Loss = True
        # NM.Add_Fea = False
        NM.Sec = [2, 3]
        # Initialise objects
        (EM, NM)=EN.Initialise_ENSim(EM, NM, FileNameE, FileNameN)
        # Check core assumptions
        EN._CheckInE(EM)
        EN._CheckInN(NM)
        # Build model
        mod = EN.build_Mod(EM, NM)        
        # Add water
        HydroPowerIn = 5
        mod = EN.add_Hydro(mod, HydroPowerIn)
        # Run model
        mod = EN.Run_Mod(mod, EM, NM)
        EN.Print_ENSim(mod, EM, NM)

EN = pyeneClass()
EN._runTests(EN)