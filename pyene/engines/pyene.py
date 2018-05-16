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
from .pyeneN import ENetworkClass as dn  # Network component
from .pyeneE import EnergyClass as de  # Energy balance/aggregation component
import json
import os


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

        return (NM, NModel, results)

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

        return (EM, EModel, results)

    # Energy and networks simulation
    def ENSim(self, FileNameE, FileNameN):
        EM = de()
        NM = dn()
        # Change assumptions
        # Load demand and RES profile
        FileName = "TimeSeries.json"
        (DemandProfiles, NoDemPeriod, BusDem, LinkDem, NoRES, NoRESP,
         LLRESType, LLRESPeriod, RESProfs, RESBus, RESLink, NoLink,
         Nohr) = self.ReadTimeS(FileName)
        # Choose a given scenario
        opt = 0
        RESProfiles = np.zeros(NoLink*Nohr, dtype=float)
        acu = 0
        for xr in range(NoRESP):
            aux1 = LLRESPeriod[LLRESType[xr][0]+opt][0]
            aux2 = LLRESPeriod[LLRESType[xr][0]+opt][1]+1
            for xs in range(aux1, aux2):
                for xt in range(Nohr):
                    RESProfiles[acu] = RESProfs[xs][xt]
                    acu += 1
        NM.settings['Demand'] = DemandProfiles[opt][:]
        NM.settings['NoTime'] = Nohr
        NM.RES['Number'] = NoRES
        NM.RES['Bus'] = RESBus
        NM.RES['Link'] = RESLink
        NM.RES['Cost'] = np.zeros(NM.RES['Number'], dtype=float)
        NM.RES['RES'] = RESProfiles

        # Initialise objects
        (EM, NM) = self.Initialise_ENSim(EM, NM, FileNameE, FileNameN)
        # Check core assumptions
        self._CheckInE(EM)
        self._CheckInN(NM)
        # Build model
        mod = self.build_Mod(EM, NM)
        # Add water
        HydroPowerIn = 300
        mod = self.add_Hydro(mod, HydroPowerIn)
        # Run model
        (mod, results) = self.Run_Mod(mod, EM, NM)

        return (mod, EM, NM)

    #                           Objective function                            #
    def OF_rule(self, m):
        return sum(sum(sum(m.vGCost[self.hGC[xh]+xg, xt] for xg in m.sGen) +
                       sum(m.vFea[self.hFea[xh]+xf, xt] for xf in m.sFea) *
                       1000000 for xt in m.sTim) * self.OFaux[xh] -
                   sum(m.ValDL[xdl]*sum(m.vGenDL[self.hDL[xh]+xdl+1, xt]
                                        for xt in m.sTim) for xdl in m.sDL) *
                   self.OFaux[xh]
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
    def Initialise_ENSim(self, EM, NM, FileNameE, FileNameN):
        # Get components of energy model
        if self.fRea:
            # Chose to load data from file
            EM.fRea = True
        else:
            FileNameE = "NoName"

        EM.initialise(FileNameE)

        # Add hydropower units to network model
        # CURRENLTY ASSIMUNG CHARACTERISTICS OF HYDRO
        NM.hydropower['Number'] = EM.size['Vectors']
        NM.hydropower['Bus'] = np.zeros(NM.hydropower['Number'], dtype=int)
        NM.hydropower['Max'] = np.zeros(NM.hydropower['Number'], dtype=int)
        NM.hydropower['Cost'] = np.zeros(NM.hydropower['Number'], dtype=float)
        for xv in range(EM.size['Vectors']):
            NM.hydropower['Bus'][xv] = xv+1
            NM.hydropower['Max'][xv] = 1000
            NM.hydropower['Cost'][xv] = 0.0001

        # Get size of network model
        NM.initialise(FileNameN)

        # Get number of required network model copies
        NoNM = (1+EM.tree['Time'][EM.size['Periods']][1] -
                EM.tree['Time'][EM.size['Periods']][0])

        NM.connections['set'] = range(NoNM)
        NM.connections['Flow'] = np.zeros(NoNM, dtype=int)
        NM.connections['Voltage'] = np.zeros(NoNM, dtype=int)
        NM.connections['Loss'] = np.zeros(NoNM, dtype=int)
        NM.connections['Generation'] = np.zeros(NoNM, dtype=int)
        NM.connections['Cost'] = np.zeros(NoNM, dtype=int)
        NM.connections['Pump'] = np.zeros(NoNM, dtype=int)
        NM.connections['Feasibility'] = np.zeros(NoNM, dtype=int)
        # Location of each copy
        for xc in NM.connections['set']:
            NM.connections['Flow'][xc] = xc*(NM.NoBranch+1)
            NM.connections['Voltage'][xc] = xc*(NM.NoBuses+1)
            NM.connections['Loss'][xc] = xc*(NM.networkE.number_of_edges()+1)
            NM.connections['Generation'][xc] = xc*(NM.generationE['Number']+1)
            NM.connections['Cost'][xc] = xc*NM.generationE['Number']
            NM.connections['Pump'][xc] = xc*(NM.pumps['Number']+1)
            NM.connections['Feasibility'][xc] = xc*NM.NoFea

        # Build LL to link the models through hydro consumption
        self.NoLL = (1+EM.tree['Time'][EM.size['Periods']][1] -
                     EM.tree['Time'][EM.size['Periods']][0])
        self.LLENM = np.zeros((self.NoLL, 2), dtype=int)
        NoGen0 = (NM.generationE['Number']-NM.hydropower['Number'] -
                  NM.RES['Number']+1)
        for xc in range(self.NoLL):
            self.LLENM[xc][:] = [EM.tree['Time'][EM.size['Periods']][0]+xc,
                                 NM.connections['Generation'][xc]+NoGen0]

        # Taking sets for modelling local objective function
        self.hGC = NM.connections['Cost']
        self.hDL = NM.connections['Pump']
        self.hFea = NM.connections['Feasibility']
        self.h = NM.connections['set']

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

    # Adding hydropower
    def add_Hydro(self, mod, HydroIn):
        aux = np.array(HydroIn)
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
        WghtAgg = EM.Weight['Node']
        self.OFaux = np.ones(len(NM.connections['set']), dtype=float)
        xp = 0
        for xn in range(EM.LL['NosBal']+1):
            aux = EM.tree['After'][xn][0]
            if aux != 0:
                for xb in range(aux, EM.tree['After'][xn][1]+1):
                    WghtAgg[xb] *= WghtAgg[xn]
            else:
                self.OFaux[xp] = WghtAgg[xn]
                xp += 1
        mod.OF = Objective(rule=self.OF_rule, sense=minimize)

        # Optimise
        opt = SolverFactory('glpk')
        # Print
        results = opt.solve(mod)

        return (mod, results)

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
        if NM.settings['Losses']:
            print('Losses are considered')
        else:
            print('Losses are neglected')
        if NM.settings['Feasibility']:
            print('Feasibility constrants are included')
        else:
            print('Feasibility constraints are neglected')
        print('Demand multiplyiers ', NM.settings['Demand'])
        print('Secuity: ', NM.settings['Security'])

    # Read time series for demand and RES
    def ReadTimeS(self, FileName):
        MODEL_JSON = os.path.join(os.path.dirname(__file__), '..\json',
                                  FileName)
        Profile_Data = json.load(open(MODEL_JSON))

        # Get number of profiles
        RESTypes = Profile_Data['metadata']['title']
        NoRESP = len(RESTypes)

        # Search for demand profile
        if "Demand" in RESTypes:
            DemandProfiles = Profile_Data['Demand']['Values']
            BusDem = Profile_Data['Demand']['Bus']
            LinkDem = Profile_Data['Demand']['Links']
            NoDemPeriod = len(Profile_Data['Demand']['Period'])
            RESTypes.remove('Demand')
            NoRESP -= 1
        else:
            DemandProfiles = [1]
            NoDemPeriod = 1
            BusDem = 'All'
            LinkDem = 0

        # Check if there is generation data
        if NoRESP == 0:
            # Remove demand data
            LLRESType = 0
            LLRESPeriod = 0
            RESProfs = 0
        else:
            # Location of each type of RES generation technology
            LLRESType = np.zeros((NoRESP, 2), dtype=int)
            NoRES = -1
            xt = -1
            sRes = np.zeros((NoRESP, 3), dtype=int)
            NoLink = 0
            for xr in RESTypes:
                xt += 1
                # Get number of periods and links
                sRes[xt][0] = len(Profile_Data[xr]['Period'])
                sRes[xt][1] = max(Profile_Data[xr]['Links'])
                sRes[xt][2] = len(Profile_Data[xr]['Bus'])
                LLRESType[xt][:] = [NoRES+1, NoRES+sRes[xt][0]]
                NoRES += sRes[xt][0]
                NoLink += sRes[xt][1]

            # Location of data for each period
            LLRESPeriod = np.zeros((NoRES+1, 2), dtype=int)
            RESBus = np.zeros(NoRES, dtype=int)
            RESLink = np.zeros(NoRES, dtype=int)
            xL = -1
            acu = -1
            for xt in range(NoRESP):
                for xp in range(sRes[xt][0]):
                    xL += 1
                    LLRESPeriod[xL][:] = [acu+1, acu+sRes[xt][1]]
                    acu += sRes[xt][1]

            # RES genertaion profiles
            if sRes[0][0]*sRes[0][1] == 1:
                Nohr = len(Profile_Data[RESTypes[0]]['Values'])
            else:
                Nohr = len(Profile_Data[RESTypes[0]]['Values'][0])
            RESProfs = np.zeros((acu+1, Nohr), dtype=float)
            acu = [0, 0, 0]
            xt = -1
            for xr in RESTypes:
                xt += 1
                aux = Profile_Data[xr]['Values']
                RESProfs[acu[0]:acu[0]+sRes[xt][0]*sRes[xt][1]][:] = aux
                acu[0] += sRes[xt][0]*sRes[xt][1]
                for xL in range(sRes[xt][2]):
                    RESBus[acu[1]] = Profile_Data[xr]['Bus'][xL]
                    RESLink[acu[1]] = Profile_Data[xr]['Links'][xL]+acu[2]
                    acu[1] += 1
                acu[2] += sRes[xt][1]

            return (DemandProfiles, NoDemPeriod, BusDem, LinkDem, NoRES,
                    NoRESP, LLRESType, LLRESPeriod, RESProfs, RESBus, RESLink,
                    NoLink, Nohr)

    # Run tests
    def _runTests(self, EN):
        # Get object
        FileNameE = "ResolutionTreeMonth01.json"
        FileNameN = "case4.json"

        # Joint simulation
        print('\n\nTESTING INTEGRATED MODEL\n\n')
        # Creat objects
        EM = de()
        NM = dn()
        # Change assumptions
        # Load demand and RES profile
#        FileName = "TimeSeries.json"
#        (DemandProfiles, NoDemPeriod, BusDem, LinkDem, NoRES, NoRESP,
#         LLRESType, LLRESPeriod, RESProfs, RESBus, RESLink, NoLink,
#         Nohr) = self.ReadTimeS(FileName)
#        # Choose a given scenario
#        opt = 0
#        RESProfiles = np.zeros(NoLink*Nohr, dtype=float)
#        acu = 0
#        for xr in range(NoRESP):
#            aux1 = LLRESPeriod[LLRESType[xr][0]+opt][0]
#            aux2 = LLRESPeriod[LLRESType[xr][0]+opt][1]+1
#            for xs in range(aux1, aux2):
#                for xt in range(Nohr):
#                    RESProfiles[acu] = RESProfs[xs][xt]
#                    acu += 1
#        NM.Settings['Demand'] = DemandProfiles[opt][:]
#        NM.Settings['NoTime'] = Nohr
#        NM.RES['Number'] = NoRES
#        NM.RES['Bus'] = RESBus
#        NM.RES['Link'] = RESLink
#        NM.RES['Cost'] = np.zeros(NM.RES['Number'], dtype=float)
#        NM.RES['RES'] = RESProfiles
        
        NM.settings['Demand'] = [1, 1.1]
        NM.settings['NoTime'] = 2
        NM.settings['Losses'] = True
        NM.settings['Security'] = [2, 3]

        # Initialise objects
        (EM, NM) = EN.Initialise_ENSim(EM, NM, FileNameE, FileNameN)
        # Check core assumptions
        EN._CheckInE(EM)
        EN._CheckInN(NM)
        # Build model
        mod = EN.build_Mod(EM, NM)
        # Add water
        HydroPowerIn = 300
        mod = EN.add_Hydro(mod, HydroPowerIn)
        # Run model
        mod = EN.Run_Mod(mod, EM, NM)
        EN.Print_ENSim(mod, EM, NM)


if __name__ == '__main__':
    EN = pyeneClass()
    EN._runTests(EN)
